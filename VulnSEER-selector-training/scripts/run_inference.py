# scripts/run_inference.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.state_formatter import StateFormatter

class SmartContextSelector:
    def __init__(self, base_model_path, lora_weights_path):
        print("[*] 加载基座模型与 RL 训练后的 LoRA...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, lora_weights_path)
        self.model.eval()

    def make_decision(self, layer_goal, candidate_item, method_code, already_selected) -> int:
        prompt = StateFormatter.build_prompt(
            layer_goal=layer_goal,
            candidate_item=candidate_item,
            method_code=method_code,
            already_selected=already_selected
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0.01,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        return 1 if generated_text == "1" else 0