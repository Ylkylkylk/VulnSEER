import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tools.state_formatter import StateFormatter
from tools.warmstart_labeler import weak_label_candidate

class WarmStartTrainer:
    """
    在 PPO 之前，先用启发式弱标签做一轮监督热启动：
    prompt -> 输出 0/1
    训练对象：policy_model.model.pretrained_model（即带 LoRA 的底层 CausalLM）
    """
    def __init__(self, policy_model, inferencer, dataset, tokenizer, config):
        self.policy_model = policy_model
        self.model = policy_model.model.pretrained_model
        self.inferencer = inferencer
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config

        ws_cfg = config.get("warmstart", {})
        self.lr = ws_cfg.get("lr", 2e-5)
        self.batch_size = ws_cfg.get("batch_size", 2)
        self.max_samples = ws_cfg.get("max_samples", 1000)

    def _build_samples(self):
        samples = []

        shuffled_dataset = list(self.dataset)
        random.shuffle(shuffled_dataset)

        for sample in shuffled_dataset:
            execution_plan = sample.get("execution_plan", [])
            cpg_path = sample.get("cpg_path")
            if not execution_plan or not cpg_path:
                continue

            self.inferencer.set_cpg_path(cpg_path)

            for layer in execution_plan[:-1]:
                current_method = layer.get("method_signature")
                if not current_method:
                    continue

                method_info = self.inferencer.get_method_context_from_cpg(current_method)
                if method_info.get("is_external", False):
                    continue

                normalized = self.inferencer.normalize_candidate_items(method_info)
                candidate_items = normalized["helpers"] + normalized["class_context"]

                if not candidate_items:
                    continue

                layer_goal = {
                    "preconditions": layer.get("preconditions", []),
                    "candidate_inputs": layer.get("candidate_inputs", [])
                }

                for item in candidate_items:
                    prompt = StateFormatter.build_prompt(
                        layer_goal=layer_goal,
                        candidate_item=item,
                        method_code=method_info.get("code", ""),
                        already_selected={"helpers": [], "class_context": []},
                        layer_index=layer.get("step_index"),
                        max_steps=self.config.get("env_baseline", {}).get("max_steps", 100)
                    )
                    label = weak_label_candidate(item)
                    samples.append((prompt, str(label)))

                    if len(samples) >= self.max_samples:
                        return samples

        return samples

    def train(self, epochs=1):
        print("[*] 开始 warm start 监督热启动...")
        samples = self._build_samples()
        if not samples:
            print("[-] warm start 未构造出任何训练样本，跳过。")
            return

        print(f"[*] warm start 样本数: {len(samples)}")

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

        self.model.train()
        device = next(self.model.parameters()).device

        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss = 0.0

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i+self.batch_size]

                prompts = [x[0] for x in batch]
                labels = [x[1] for x in batch]

                full_texts = [p + l for p, l in zip(prompts, labels)]

                enc = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config["model"]["max_context_length"]
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                labels_tensor = input_ids.clone()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_tensor
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, (len(samples) + self.batch_size - 1) // self.batch_size)
            print(f"[warmstart] epoch={epoch+1} avg_loss={avg_loss:.4f}")

        print("[+] warm start 完成。")