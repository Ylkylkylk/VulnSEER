import os
import json
import random
from typing import Dict, Any, List

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ModuleNotFoundError:
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from peft import PeftModel
except ModuleNotFoundError:
    PeftModel = None

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

from tools.state_formatter import StateFormatter


def _require_dependency(name: str, value):
    if value is None:
        raise ImportError(f"Policy selector requires the Python package {name}; install it before using this policy.")


def build_codebert_selector_text(
    layer_goal: Dict[str, Any],
    candidate_item: Dict[str, Any],
    method_code: str,
    already_selected: Dict[str, Any]
) -> str:
    """
    Shared input-construction function for CodeBERT and Qwen-LoRA binary selectors.

    Note:
    To compare the CodeBERT baseline and Qwen3-8B-LoRA baseline fairly,
    both training and inference should use the same text template.
    """
    candidate_source_type = candidate_item.get("source_type", "")
    candidate_signature = candidate_item.get("signature", "")
    candidate_content = candidate_item.get("content", "")

    layer_goal_text = json.dumps(layer_goal, ensure_ascii=False)
    already_selected_text = json.dumps(already_selected, ensure_ascii=False)

    return (
        f"Current Code:\n{method_code}\n\n"
        f"Layer Goal:\n{layer_goal_text}\n\n"
        f"Already Selected:\n{already_selected_text}\n\n"
        f"Candidate Type:\n{candidate_source_type}\n\n"
        f"Candidate Signature:\n{candidate_signature}\n\n"
        f"Candidate Content:\n{candidate_content}"
    )


def _sanitize_no_split_modules(model):
    """
    Normalize model._no_split_modules into List[str].
    This prevents nested set / tuple / list values from triggering accelerate errors such as:
    TypeError: unhashable type: 'set'
    """
    if model is None or not hasattr(model, "_no_split_modules"):
        return

    raw = getattr(model, "_no_split_modules", None)
    if raw is None:
        return

    cleaned: List[str] = []

    def _flatten(x):
        if x is None:
            return
        if isinstance(x, str):
            cleaned.append(x)
        elif isinstance(x, (set, list, tuple)):
            for y in x:
                _flatten(y)
        else:
            cleaned.append(str(x))

    _flatten(raw)

    # Deduplicate while preserving order.
    model._no_split_modules = list(dict.fromkeys(cleaned))


class BaseOrLoRASelector:
    """
    Original RL / base selector.

    This selector uses the CausalLM form:
    - Input: StateFormatter.build_prompt(...)
    - generate(max_new_tokens=1)
    - Generate "1" for include and "0" for exclude.

    Note:
    This class is not suitable for Qwen3-LoRA models trained with SequenceClassification.
    If Qwen3-8B was fine-tuned as a binary LoRA classifier, use QwenLoRASequenceClassificationSelector.
    """

    def __init__(self, base_model_path: str, lora_path: str = None):
        _require_dependency("torch", torch)
        _require_dependency("transformers", AutoTokenizer)
        _require_dependency("transformers", AutoModelForCausalLM)
        if lora_path:
            _require_dependency("peft", PeftModel)

        print("[*] Loading CausalLM decision model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if not torch.cuda.is_available():
            base_model = base_model.to("cpu")

        # Important: sanitize no_split_modules before loading LoRA.
        _sanitize_no_split_modules(base_model)
        _sanitize_no_split_modules(getattr(base_model, "model", None))
        _sanitize_no_split_modules(getattr(getattr(base_model, "base_model", None), "model", None))

        # Optional debug output.
        print("[debug] base_model._no_split_modules =", getattr(base_model, "_no_split_modules", None))

        if lora_path:
            print(f"[*] Loading LoRA weights: {lora_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_path)

            # Sanitize again in case adapter wrapping reintroduces invalid values.
            _sanitize_no_split_modules(self.model)
            _sanitize_no_split_modules(getattr(self.model, "model", None))
        else:
            print("[*] LoRA is not loaded; using the base model only.")
            self.model = base_model

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def make_decision(
        self,
        layer_goal: Dict[str, Any],
        candidate_item: Dict[str, Any],
        method_code: str,
        already_selected: Dict[str, Any]
    ) -> int:
        prompt = StateFormatter.build_prompt(
            layer_goal=layer_goal,
            candidate_item=candidate_item,
            method_code=method_code,
            already_selected=already_selected,
        )

        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        return 1 if generated_text == "1" else 0


class CodeBERTBinarySelector:
    """
    Original CodeBERT binary selector.

    Output rule:
    - softmax(logits)[1] >= threshold => include
    - otherwise exclude
    """

    def __init__(self, model_path: str, max_length: int = 512, threshold: float = 0.5):
        _require_dependency("torch", torch)
        _require_dependency("transformers", AutoTokenizer)
        _require_dependency("transformers", AutoModelForSequenceClassification)

        print(f"[*] Loading CodeBERT selector: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.threshold = threshold

    def make_decision(
        self,
        layer_goal: Dict[str, Any],
        candidate_item: Dict[str, Any],
        method_code: str,
        already_selected: Dict[str, Any]
    ) -> int:
        text = build_codebert_selector_text(
            layer_goal=layer_goal,
            candidate_item=candidate_item,
            method_code=method_code,
            already_selected=already_selected,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            include_prob = probs[0, 1].item()

        return 1 if include_prob >= self.threshold else 0


class QwenLoRASequenceClassificationSelector:
    """
    Qwen3-8B binary selector after LoRA fine-tuning.

    Suitable when:
    - Qwen3-8B was trained with AutoModelForSequenceClassification + LoRA.
    - The data format matches the CodeBERT selector: text -> label(0/1).
    - Inference uses softmax(logits)[1] as the include probability.

    Not suitable when:
    - Qwen LoRA was trained as a generative CausalLM; use BaseOrLoRASelector in that case.
    """

    def __init__(
        self,
        base_model_path: str,
        lora_path: str,
        max_length: int = 1024,
        threshold: float = 0.5,
        load_in_4bit: bool = True,
        bf16: bool = True,
    ):
        if not base_model_path:
            raise ValueError("QwenLoRASequenceClassificationSelector requires base_model_path")
        if not lora_path:
            raise ValueError("QwenLoRASequenceClassificationSelector requires lora_path")
        _require_dependency("torch", torch)
        _require_dependency("transformers", AutoTokenizer)
        _require_dependency("transformers", AutoModelForSequenceClassification)
        _require_dependency("transformers", BitsAndBytesConfig)
        _require_dependency("peft", PeftModel)

        print("[*] Loading Qwen3 LoRA SequenceClassification selector")
        print(f"[*] base_model_path = {base_model_path}")
        print(f"[*] lora_path       = {lora_path}")
        print(f"[*] threshold       = {threshold}")
        print(f"[*] max_length      = {max_length}")
        print(f"[*] load_in_4bit    = {load_in_4bit}")

        self.max_length = max_length
        self.threshold = threshold

        # Prefer the LoRA directory if it contains a tokenizer; otherwise use the base-model directory.
        tokenizer_path = (
            lora_path
            if os.path.exists(os.path.join(lora_path, "tokenizer_config.json"))
            else base_model_path
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if bf16 else torch.float16
        else:
            compute_dtype = torch.float32

        model_kwargs = {
            "num_labels": 2,
            "trust_remote_code": True,
            "dtype": compute_dtype,
        }

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        if load_in_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        print("[*] Loading Qwen3 SequenceClassification base model...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            **model_kwargs,
        )

        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        base_model.config.use_cache = False

        # Sanitize no_split_modules to avoid accelerate errors such as unhashable type: set.
        _sanitize_no_split_modules(base_model)
        _sanitize_no_split_modules(getattr(base_model, "model", None))
        _sanitize_no_split_modules(getattr(getattr(base_model, "base_model", None), "model", None))

        print("[*] Loading Qwen3 classification LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        _sanitize_no_split_modules(self.model)
        _sanitize_no_split_modules(getattr(self.model, "model", None))

        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")

        self.model.eval()

        # Print trainable parameters once to confirm inference is not accidentally enabling training.
        try:
            self.model.print_trainable_parameters()
        except Exception:
            pass

    def make_decision(
        self,
        layer_goal: Dict[str, Any],
        candidate_item: Dict[str, Any],
        method_code: str,
        already_selected: Dict[str, Any]
    ) -> int:
        text = build_codebert_selector_text(
            layer_goal=layer_goal,
            candidate_item=candidate_item,
            method_code=method_code,
            already_selected=already_selected,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # With device_map="auto", place inputs on the device of the first model parameter.
        # This is usually sufficient for single-GPU 4-bit loading.
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            include_prob = probs[0, 1].item()

        return 1 if include_prob >= self.threshold else 0


class GPTSelector:
    def __init__(
        self,
        model_name: str = "gpt-5.2",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = None,
        max_retries: int = 3,
    ):
        print(f"[*] Loading GPT selector: {model_name}")
        _require_dependency("openai", OpenAI)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable not found: {api_key_env}")

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def make_decision(
        self,
        layer_goal: Dict[str, Any],
        candidate_item: Dict[str, Any],
        method_code: str,
        already_selected: Dict[str, Any]
    ) -> int:
        prompt = (
            "You are a binary selector for vulnerability reasoning context selection.\n"
            "Return exactly one character: 1 or 0.\n"
            "1 means include this candidate item.\n"
            "0 means exclude this candidate item.\n\n"
            "Prefer including information that directly helps satisfy constraints, instantiate variables, recover fields/constants/helpers, or build payload-relevant objects.\n"
            "Exclude redundant, generic, irrelevant, or obviously low-value context.\n\n"
            f"Layer Goal:\n{json.dumps(layer_goal, ensure_ascii=False)}\n\n"
            f"Candidate Item:\n{json.dumps(candidate_item, ensure_ascii=False)}\n\n"
            f"Already Selected:\n{json.dumps(already_selected, ensure_ascii=False)}\n\n"
            f"Current Method Code:\n{method_code}\n"
        )

        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.responses.create(model=self.model_name, input=prompt)
                text = (resp.output_text or "").strip()
                if text == "1" or text.startswith("1"):
                    return 1
                if text == "0" or text.startswith("0"):
                    return 0
                last_err = ValueError(f"GPT returned an invalid result: {text!r}")
            except Exception as e:
                last_err = e

        print(f"[WARN] GPT selector failed; defaulting to exclude. err={last_err}")
        return 0


class AlwaysIncludeSelector:
    def make_decision(self, layer_goal, candidate_item, method_code, already_selected) -> int:
        return 1


class AlwaysExcludeSelector:
    def make_decision(self, layer_goal, candidate_item, method_code, already_selected) -> int:
        return 0


class RandomSelector:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def make_decision(self, layer_goal, candidate_item, method_code, already_selected) -> int:
        return self.rng.randint(0, 1)


def build_selector(
    policy: str,
    base_model: str = None,
    lora_path: str = None,
    seed: int = 42,
    codebert_model_path: str = None,
    codebert_threshold: float = 0.5,
    gpt_model_name: str = "gpt-5.2",
    gpt_base_url: str = None,
    qwen_lora_threshold: float = 0.5,
    qwen_lora_max_length: int = 1024,
    qwen_lora_load_in_4bit: bool = True,
):
    """
    Selector construction entry point.

    Available policies:
    - rl              : CausalLM + LoRA; generates 0/1 decisions.
    - base            : CausalLM base model; generates 0/1 decisions.
    - codebert        : CodeBERT binary selector.
    - qwen_lora       : Qwen3-8B + LoRA binary selector.
    - gpt             : GPT API selector
    - always_include  : Always include.
    - always_exclude  : Always exclude.
    - random          : Randomly include or exclude.
    """

    if policy == "rl":
        if not base_model or not lora_path:
            raise ValueError("policy=rl requires --base_model and --lora_path")
        return BaseOrLoRASelector(base_model_path=base_model, lora_path=lora_path)

    if policy == "base":
        if not base_model:
            raise ValueError("policy=base requires --base_model")
        return BaseOrLoRASelector(base_model_path=base_model, lora_path=None)

    if policy == "codebert":
        if not codebert_model_path:
            raise ValueError("policy=codebert requires --codebert_model_path")
        return CodeBERTBinarySelector(
            model_path=codebert_model_path,
            threshold=codebert_threshold,
        )

    if policy == "qwen_lora":
        if not base_model or not lora_path:
            raise ValueError("policy=qwen_lora requires --base_model and --lora_path")
        return QwenLoRASequenceClassificationSelector(
            base_model_path=base_model,
            lora_path=lora_path,
            threshold=qwen_lora_threshold,
            max_length=qwen_lora_max_length,
            load_in_4bit=qwen_lora_load_in_4bit,
        )

    if policy == "gpt":
        return GPTSelector(model_name=gpt_model_name, base_url=gpt_base_url)

    if policy == "always_include":
        return AlwaysIncludeSelector()

    if policy == "always_exclude":
        return AlwaysExcludeSelector()

    if policy == "random":
        return RandomSelector(seed=seed)

    raise ValueError(f"Unknown policy: {policy}")
