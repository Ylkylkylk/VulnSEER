import os
import json
import time
import random
import argparse
from typing import Dict, Any, List

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel
from openai import OpenAI

from bottom_up_inferencer import LayerWiseExploitGoalInferencer, ExploitStateResolver
from tools.joern_slicer import JoernSlicer
from tools.payload_evaluator import ExploitPayloadEvaluator
from tools.context_fallback import ContextFallbackBuilder
from tools.state_formatter import StateFormatter


# =========================================================
# Utilities
# =========================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> list:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_low_value_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    markers = [
        "<empty>",
        "fallback",
        "assumed to",
        "generic context",
        "method not found",
        "timeout",
        "no direct field usage found",
    ]
    return any(m in lowered for m in markers)


def has_external_layer(sample: Dict[str, Any]) -> bool:
    chain = sample.get("execution_plan", [])
    for layer in chain:
        sig = layer.get("method_signature", "")
        if sig.startswith("org.apache.") or sig.startswith("java.") or sig.startswith("javax."):
            return True
    return False


def build_codebert_selector_text(
    layer_goal: Dict[str, Any],
    candidate_item: Dict[str, Any],
    method_code: str,
    already_selected: Dict[str, Any]
) -> str:
    candidate_source_type = candidate_item.get("source_type", "")
    candidate_signature = candidate_item.get("signature", "")
    candidate_content = candidate_item.get("content", "")

    layer_goal_text = json.dumps(layer_goal, ensure_ascii=False)
    already_selected_text = json.dumps(already_selected, ensure_ascii=False)

    text = (
        f"Current Code:\n{method_code}\n\n"
        f"Layer Goal:\n{layer_goal_text}\n\n"
        f"Already Selected:\n{already_selected_text}\n\n"
        f"Candidate Type:\n{candidate_source_type}\n\n"
        f"Candidate Signature:\n{candidate_signature}\n\n"
        f"Candidate Content:\n{candidate_content}"
    )
    return text


# =========================================================
# Policy Selectors
# =========================================================

class BaseOrLoRASelector:
    def __init__(self, base_model_path: str, lora_path: str = None):
        print("[*] 加载 CausalLM 决策模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if not torch.cuda.is_available():
            base_model = base_model.to("cpu")

        if lora_path:
            print(f"[*] 加载 LoRA 权重: {lora_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print("[*] 不加载 LoRA，仅使用基座模型。")
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
            already_selected=already_selected
        )

        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        return 1 if generated_text == "1" else 0


class CodeBERTBinarySelector:
    def __init__(self, model_path: str, max_length: int = 512, threshold: float = 0.5):
        print(f"[*] 加载 CodeBERT selector: {model_path}")
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
            already_selected=already_selected
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

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
        max_retries: int = 3
    ):
        print(f"[*] 加载 GPT selector: {model_name}")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"未找到环境变量 {api_key_env}")

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

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
            "Prefer including information that directly helps satisfy constraints, "
            "instantiate variables, recover fields/constants/helpers, or build payload-relevant objects.\n"
            "Exclude redundant, generic, irrelevant, or obviously low-value context.\n\n"
            f"Layer Goal:\n{json.dumps(layer_goal, ensure_ascii=False)}\n\n"
            f"Candidate Item:\n{json.dumps(candidate_item, ensure_ascii=False)}\n\n"
            f"Already Selected:\n{json.dumps(already_selected, ensure_ascii=False)}\n\n"
            f"Current Method Code:\n{method_code}\n"
        )

        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.responses.create(
                    model=self.model_name,
                    input=prompt
                )
                text = (resp.output_text or "").strip()

                if text == "1":
                    return 1
                if text == "0":
                    return 0
                if text.startswith("1"):
                    return 1
                if text.startswith("0"):
                    return 0

                last_err = ValueError(f"GPT 返回非法结果: {text!r}")
            except Exception as e:
                last_err = e

        print(f"[WARN] GPT selector 调用失败，默认 exclude。err={last_err}")
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


# =========================================================
# Evaluation Core
# =========================================================

class EvaluatorRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        selector,
        use_llm_judge: bool = False,
        max_steps_override: int = None,
    ):
        self.config = config
        self.selector = selector

        cpg_path = config["env_baseline"]["cpg_path"]
        joern_bin = config["env_baseline"]["joern_bin"]

        self.inferencer = LayerWiseExploitGoalInferencer(
            cpg_path=cpg_path,
            joern_bin=joern_bin,
            model=config["solver_llm"]["model_name"],
            debug_prompt_dir=config["paths"]["debug_prompt_dir"]
        )
        self.solver = ExploitStateResolver(self.inferencer)

        self.slicer = JoernSlicer(
            cpg_path=cpg_path,
            joern_bin=joern_bin,
            temp_dir=config["paths"]["temp_dir"]
        )

        self.evaluator = ExploitPayloadEvaluator(use_llm_judge=use_llm_judge)
        self.context_fallback = ContextFallbackBuilder()

        self.max_steps_per_episode = (
            max_steps_override
            if max_steps_override is not None
            else config.get("env_baseline", {}).get("max_steps", 100)
        )

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = sample.get("sample_id", "unknown_sample")
        project_id = sample.get("project_id", "unknown_project")
        cpg_path = sample.get("cpg_path")
        execution_plan = sample["execution_plan"]
        gt_payload = sample["ground_truth_payload"]

        if not cpg_path:
            raise ValueError(f"sample {sample_id} 缺少 cpg_path")

        self.solver.set_cpg_path(cpg_path)
        self.slicer.set_cpg_path(cpg_path)

        start_time = time.time()

        trace = []
        include_count = 0
        exclude_count = 0
        fallback_include_count = 0
        empty_include_count = 0
        external_step_count = 0

        final_payload = {}
        final_score = 0.0
        is_success = False
        truncated = False
        total_steps = 0

        try:
            gen = self.solver.resolve_execution_plan(execution_plan)
            event = next(gen)

            while True:
                if event.get("is_done", False):
                    final_payload = event.get("final_payload", {})
                    if final_payload:
                        final_score = self.evaluator.evaluate(final_payload, gt_payload)
                    else:
                        final_score = 0.0
                    is_success = final_score >= 0.5
                    break

                total_steps += 1
                if total_steps > self.max_steps_per_episode:
                    truncated = True
                    final_payload = event.get("final_payload", {})
                    final_score = 0.0
                    is_success = False
                    break

                candidate_item = event["candidate_item"]
                current_method = event["current_method"]
                layer_goal = event["constraint"]
                method_code = event["method_code"]
                already_selected = event.get("already_selected", {})

                if event.get("is_external", False):
                    external_step_count += 1

                action = self.selector.make_decision(
                    layer_goal=layer_goal,
                    candidate_item=candidate_item,
                    method_code=method_code,
                    already_selected=already_selected
                )

                selected_text = ""
                selection_mode = "exclude"

                if action == 1:
                    sliced_text = self.slicer.slice_context_item(
                        current_method=current_method,
                        item=candidate_item
                    )
                    if sliced_text and "not found" not in sliced_text.lower() and "timeout" not in sliced_text.lower():
                        selected_text = sliced_text
                        selection_mode = "slice"
                    else:
                        selected_text = self.context_fallback.generate_fallback_for_context_item(candidate_item)
                        selection_mode = "fallback"
                        fallback_include_count += 1

                    if is_low_value_text(selected_text):
                        empty_include_count += 1

                    include_count += 1
                else:
                    exclude_count += 1

                trace.append({
                    "step": total_steps,
                    "method": current_method,
                    "candidate_signature": candidate_item.get("signature", ""),
                    "candidate_name": candidate_item.get("name", ""),
                    "candidate_source_type": candidate_item.get("source_type", "unknown"),
                    "action": action,
                    "selection_mode": selection_mode,
                    "selected_text_preview": selected_text[:200] if selected_text else "",
                })

                solver_feedback = {
                    "action": action,
                    "selected_text": selected_text,
                    "source_type": candidate_item.get("source_type", "unknown")
                }

                event = gen.send(solver_feedback)

        except StopIteration:
            pass
        except Exception as e:
            final_score = 0.0
            is_success = False
            trace.append({
                "step": total_steps,
                "error": str(e)
            })

        elapsed = time.time() - start_time

        result = {
            "sample_id": sample_id,
            "project_id": project_id,
            "chain_length": len(execution_plan),
            "has_external_layer": has_external_layer(sample),
            "num_decisions": include_count + exclude_count,
            "num_include": include_count,
            "num_exclude": exclude_count,
            "fallback_include_count": fallback_include_count,
            "low_value_include_count": empty_include_count,
            "external_step_count": external_step_count,
            "payload_score": round(float(final_score), 6),
            "is_payload_success": bool(is_success),
            "truncated": bool(truncated),
            "elapsed_sec": round(elapsed, 4),
            "final_payload": final_payload,
            "trace": trace,
        }
        return result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    avg_score = sum(r["payload_score"] for r in results) / n
    success_rate = sum(1 for r in results if r["is_payload_success"]) / n
    avg_include = sum(r["num_include"] for r in results) / n
    avg_exclude = sum(r["num_exclude"] for r in results) / n
    avg_steps = sum(r["num_decisions"] for r in results) / n
    avg_elapsed = sum(r["elapsed_sec"] for r in results) / n

    total_include = sum(r["num_include"] for r in results)
    total_low_value_include = sum(r["low_value_include_count"] for r in results)
    total_fallback_include = sum(r["fallback_include_count"] for r in results)

    low_value_include_rate = total_low_value_include / max(1, total_include)
    fallback_include_rate = total_fallback_include / max(1, total_include)

    external_samples = [r for r in results if r["has_external_layer"]]
    non_external_samples = [r for r in results if not r["has_external_layer"]]

    def avg_score_of(xs):
        if not xs:
            return None
        return sum(x["payload_score"] for x in xs) / len(xs)

    summary = {
        "num_samples": n,
        "avg_payload_score": round(avg_score, 6),
        "success_rate": round(success_rate, 6),
        "avg_include_count": round(avg_include, 6),
        "avg_exclude_count": round(avg_exclude, 6),
        "avg_steps": round(avg_steps, 6),
        "avg_elapsed_sec": round(avg_elapsed, 6),
        "low_value_include_rate": round(low_value_include_rate, 6),
        "fallback_include_rate": round(fallback_include_rate, 6),
        "external_avg_payload_score": None if avg_score_of(external_samples) is None else round(avg_score_of(external_samples), 6),
        "non_external_avg_payload_score": None if avg_score_of(non_external_samples) is None else round(avg_score_of(non_external_samples), 6),
    }
    return summary


# =========================================================
# Main
# =========================================================

def build_selector(
    policy: str,
    base_model: str = None,
    lora_path: str = None,
    seed: int = 42,
    codebert_model_path: str = None,
    codebert_threshold: float = 0.5,
    gpt_model_name: str = "gpt-5.2",
    gpt_base_url: str = None,
):
    if policy == "rl":
        if not base_model or not lora_path:
            raise ValueError("policy=rl 时必须提供 --base_model 和 --lora_path")
        return BaseOrLoRASelector(base_model_path=base_model, lora_path=lora_path)

    if policy == "base":
        if not base_model:
            raise ValueError("policy=base 时必须提供 --base_model")
        return BaseOrLoRASelector(base_model_path=base_model, lora_path=None)

    if policy == "codebert":
        if not codebert_model_path:
            raise ValueError("policy=codebert 时必须提供 --codebert_model_path")
        return CodeBERTBinarySelector(
            model_path=codebert_model_path,
            threshold=codebert_threshold
        )

    if policy == "gpt":
        return GPTSelector(
            model_name=gpt_model_name,
            base_url=gpt_base_url
        )

    if policy == "always_include":
        return AlwaysIncludeSelector()

    if policy == "always_exclude":
        return AlwaysExcludeSelector()

    if policy == "random":
        return RandomSelector(seed=seed)

    raise ValueError(f"未知 policy: {policy}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate context-selection policy on vulnerability reasoning dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--policy",
        type=str,
        choices=["rl", "base", "codebert", "gpt", "always_include", "always_exclude", "random"],
        required=True
    )
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)

    parser.add_argument("--codebert_model_path", type=str, default=None)
    parser.add_argument("--codebert_threshold", type=float, default=0.5)

    parser.add_argument("--gpt_model_name", type=str, default="gpt-5.2")
    parser.add_argument("--gpt_base_url", type=str, default=None)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--use_llm_judge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    config = load_config(args.config)
    dataset = load_dataset(args.dataset)

    selector = build_selector(
        policy=args.policy,
        base_model=args.base_model,
        lora_path=args.lora_path,
        seed=args.seed,
        codebert_model_path=args.codebert_model_path,
        codebert_threshold=args.codebert_threshold,
        gpt_model_name=args.gpt_model_name,
        gpt_base_url=args.gpt_base_url,
    )

    runner = EvaluatorRunner(
        config=config,
        selector=selector,
        use_llm_judge=args.use_llm_judge,
        max_steps_override=args.max_steps
    )

    all_results = []
    for idx, sample in enumerate(dataset, 1):
        print(f"\n{'=' * 60}")
        print(f"[*] Evaluating sample {idx}/{len(dataset)}: {sample.get('sample_id', 'unknown')}")
        print(f"{'=' * 60}")

        result = runner.evaluate_sample(sample)
        all_results.append(result)

        print(
            f"[RESULT] sample_id={result['sample_id']} | "
            f"score={result['payload_score']:.2f} | "
            f"success={result['is_payload_success']} | "
            f"include={result['num_include']} | "
            f"exclude={result['num_exclude']} | "
            f"steps={result['num_decisions']} | "
            f"truncated={result['truncated']}"
        )

    summary = summarize_results(all_results)

    output_obj = {
        "policy": args.policy,
        "dataset": args.dataset,
        "selector_config": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "codebert_model_path": args.codebert_model_path,
            "codebert_threshold": args.codebert_threshold,
            "gpt_model_name": args.gpt_model_name,
            "gpt_base_url": args.gpt_base_url,
        },
        "summary": summary,
        "results": all_results
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("[+] Evaluation Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[+] 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()