import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead

class PolicyModel:
    def __init__(self, config: dict):
        self.model_path = os.environ.get(
            "VULNSEER_SELECTOR_BASE_MODEL",
            config["model"]["base_model_path"],
        )
        self.lora_config_dict = config["lora"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[*] 正在以 4-bit 量化模式加载基座模型: {self.model_path} ...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.base_model.config.use_cache = False
        self.base_model.gradient_checkpointing_enable()
        self.base_model = prepare_model_for_kbit_training(self.base_model)

        self._inject_lora()

        if not hasattr(self.model, "pretrained_model"):
            raise RuntimeError("ValueHead 模型缺少 pretrained_model，TRL PPOTrainer 无法正常工作。")

        if not hasattr(self.model, "is_peft_model"):
            self.model.is_peft_model = True
    
    def iter_trainable_named_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def _inject_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config_dict.get("r", 16),
            lora_alpha=self.lora_config_dict.get("lora_alpha", 32),
            target_modules=self.lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=self.lora_config_dict.get("lora_dropout", 0.05),
            bias=self.lora_config_dict.get("bias", "none")
        )

        peft_model = get_peft_model(self.base_model, lora_config)
        peft_model.print_trainable_parameters()

        # 兼容当前 TRL PPOTrainer
        peft_model.is_peft_model = True

        self.model = AutoModelForCausalLMWithValueHead(peft_model)

        # wrapper 也补齐属性
        self.model.is_peft_model = True

        # 某些版本的 TRL 会访问 active_peft_config
        if not hasattr(self.model.pretrained_model, "active_peft_config") and hasattr(peft_model, "active_peft_config"):
            self.model.pretrained_model.active_peft_config = peft_model.active_peft_config

        self.model.v_head.requires_grad_(True)

        trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"[*] trainable parameter tensors: {len(trainable_names)}")
        print("[*] first few trainable params:")
        for n in trainable_names[:20]:
            print("   ", n)

        print("[+] QLoRA + ValueHead 适配器注入完成，准备进入强化学习。")

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer
