# trainer/skyrl_trainer.py
import torch
import os
import random
from collections import Counter
from trl import PPOTrainer, PPOConfig

from .curriculum_manager import CurriculumManager
from envs.vuln_reasoning_env import VulnReasoningEnv

class VulnRLTrainer:
    def __init__(self, env, policy_model, default_config, curriculum_config_path,
                 val_dataset=None, inferencer=None, joern_slicer=None):
        self.env = env
        self.model, self.tokenizer = policy_model.get_model_and_tokenizer()
        self.curriculum = CurriculumManager(curriculum_config_path, self.env)

        self.default_config = default_config
        self.val_dataset = val_dataset or []
        self.inferencer = inferencer
        self.joern_slicer = joern_slicer

        ppo_config = default_config.get("skyrl_ppo", {})

        self.rollout_batch_size = max(2, ppo_config.get("batch_size", 8))
        self.mini_batch_size = ppo_config.get("mini_batch_size", 2)

        self.invalid_action_penalty = ppo_config.get("invalid_action_penalty", 0.1)
        self.generation_temperature = ppo_config.get("temperature", 0.3)
        self.save_every_updates = ppo_config.get("save_every_updates", 20)
        self.eval_every_updates = ppo_config.get("eval_every_updates", 20)

        config = PPOConfig(
            learning_rate=ppo_config.get("learning_rate", 5e-6),
            batch_size=self.rollout_batch_size,
            mini_batch_size=self.mini_batch_size,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            seed=42
        )

        self.ppo_trainer = PPOTrainer(
            config=config,
            model=self.model,
            ref_model=None,
            tokenizer=self.tokenizer
        )

        self.query_buffer = []
        self.response_buffer = []
        self.reward_buffer = []

        self.action_counter = Counter()
        self.episode_reward = 0.0
        self.episode_include = 0
        self.episode_exclude = 0
        self.episode_invalid = 0
        self.episode_selected_tokens = 0
        self.episode_low_value_include = 0

        self.update_counter = 0

        self.best_eval_score = float("-inf")
        self.no_improve_rounds = 0
        self.best_ckpt_dir = os.path.join(default_config["paths"]["output_root"], "best_checkpoint")
        os.makedirs(self.best_ckpt_dir, exist_ok=True)

    def _snapshot_trainable_params(self, max_items=8):
        snap = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and ("lora_" in name or "v_head" in name):
                snap[name] = param.detach().float().view(-1)[:4].cpu().clone()
                if len(snap) >= max_items:
                    break
        return snap

    def _compare_param_snapshots(self, before, after):
        deltas = {}
        for k in before:
            if k in after:
                deltas[k] = (after[k] - before[k]).abs().mean().item()
        return deltas

    def _collect_grad_stats(self):
        grad_count = 0
        grad_norm_sum = 0.0
        lora_grad_count = 0
        lora_grad_norm_sum = 0.0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                continue

            g = param.grad.detach().float().norm().item()
            grad_count += 1
            grad_norm_sum += g

            if "lora_" in name or "v_head" in name:
                lora_grad_count += 1
                lora_grad_norm_sum += g

        return {
            "grad_count": grad_count,
            "grad_norm_sum": grad_norm_sum,
            "lora_grad_count": lora_grad_count,
            "lora_grad_norm_sum": lora_grad_norm_sum,
        }

    def _flush_rollout_if_ready(self):
        if len(self.query_buffer) < self.rollout_batch_size:
            return None

        before = self._snapshot_trainable_params()

        stats = self.ppo_trainer.step(
            self.query_buffer,
            self.response_buffer,
            self.reward_buffer
        )

        grad_stats = self._collect_grad_stats()
        after = self._snapshot_trainable_params()
        param_deltas = self._compare_param_snapshots(before, after)

        self.update_counter += 1
        print(
            f"[DEBUG:ppo_update] update={self.update_counter} | "
            f"grad_count={grad_stats['grad_count']} | "
            f"lora_grad_count={grad_stats['lora_grad_count']} | "
            f"grad_norm_sum={grad_stats['grad_norm_sum']:.6f} | "
            f"lora_grad_norm_sum={grad_stats['lora_grad_norm_sum']:.6f}"
        )

        for name, delta in list(param_deltas.items())[:5]:
            print(f"[DEBUG:param_delta] {name}: {delta:.8f}")

        self.query_buffer.clear()
        self.response_buffer.clear()
        self.reward_buffer.clear()

        if self.update_counter % self.eval_every_updates == 0 and self.val_dataset and self.inferencer and self.joern_slicer:
            eval_metrics = self.evaluate_policy(num_episodes=min(10, len(self.val_dataset)))
            eval_score = eval_metrics["avg_payload_score"]
            print(f"[EVAL] {eval_metrics}")

            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                self.no_improve_rounds = 0
                self.model.pretrained_model.save_pretrained(self.best_ckpt_dir)
                self.tokenizer.save_pretrained(self.best_ckpt_dir)
                print(f"[+] 保存 best checkpoint 到 {self.best_ckpt_dir}")
            else:
                self.no_improve_rounds += 1

        return stats

    def _act(self, obs_text):
        device = self.model.pretrained_model.device
        query_tensor = self.tokenizer.encode(obs_text, return_tensors="pt").to(device)[0]

        generation_kwargs = {
            "max_new_tokens": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": self.generation_temperature,
            "top_k": 2
        }

        response_tensors = self.ppo_trainer.generate(
            [query_tensor],
            return_prompt=False,
            **generation_kwargs
        )
        response_tensor = response_tensors[0]

        action_str = self.tokenizer.decode(response_tensor, skip_special_tokens=True).strip()

        invalid_action = False
        if action_str == "1":
            action = 1
        elif action_str == "0":
            action = 0
        else:
            action = 0
            invalid_action = True

        if getattr(self.env, "mask_action_0", False) and action == 0:
            action = 1
            invalid_action = False

        return query_tensor, response_tensor, action, invalid_action

    @torch.no_grad()
    def evaluate_policy(self, num_episodes=10):
        eval_env = VulnReasoningEnv(
            config=self.default_config,
            exploit_state_resolver=type(self.env.solver)(self.inferencer),
            joern_slicer=self.joern_slicer,
            dataset=random.sample(self.val_dataset, min(num_episodes, len(self.val_dataset))),
            tokenizer=self.tokenizer,
            use_llm_judge=False
        )

        total_score = 0.0
        total_success = 0
        total_include = 0
        total_exclude = 0
        total_rewards = 0.0
        total_episodes = 0

        obs_text, _ = eval_env.reset()

        while total_episodes < num_episodes:
            device = self.model.pretrained_model.device
            query_tensor = self.tokenizer.encode(obs_text, return_tensors="pt").to(device)[0]

            response_tensors = self.ppo_trainer.generate(
                [query_tensor],
                return_prompt=False,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
            response_tensor = response_tensors[0]
            action_str = self.tokenizer.decode(response_tensor, skip_special_tokens=True).strip()

            if action_str == "1":
                action = 1
            else:
                action = 0

            next_obs_text, reward, is_done, truncated, info = eval_env.step(action)

            total_rewards += reward
            if action == 1:
                total_include += 1
            else:
                total_exclude += 1

            if is_done or truncated:
                total_score += float(info.get("payload_score", 0.0) or 0.0)
                total_success += 1 if info.get("is_payload_success", False) else 0
                total_episodes += 1
                if total_episodes < num_episodes:
                    obs_text, _ = eval_env.reset()
            else:
                obs_text = next_obs_text

        return {
            "num_eval_episodes": total_episodes,
            "avg_payload_score": total_score / max(1, total_episodes),
            "success_rate@0.8": total_success / max(1, total_episodes),
            "avg_include_count_proxy": total_include / max(1, total_episodes),
            "avg_exclude_count_proxy": total_exclude / max(1, total_episodes),
            "avg_reward": total_rewards / max(1, total_episodes)
        }

    def train(self, total_timesteps: int):
        print("[+] 🚀 开始基于 Hugging Face TRL 的强化学习微调...")

        obs_text, info = self.env.reset()

        for step in range(total_timesteps):
            query_tensor, response_tensor, action, invalid_action = self._act(obs_text)

            next_obs_text, reward, is_done, truncated, info = self.env.step(action)

            if invalid_action:
                reward -= self.invalid_action_penalty

            device = self.model.pretrained_model.device
            reward_tensor = torch.tensor(reward, dtype=torch.float, device=device)

            self.action_counter[action] += 1
            if invalid_action:
                self.action_counter["invalid"] += 1
                self.episode_invalid += 1

            self.episode_reward += reward
            if action == 1:
                self.episode_include += 1
            else:
                self.episode_exclude += 1

            self.episode_selected_tokens += info.get("selected_text_token_count", 0)
            if action == 1 and info.get("selected_text_is_low_value", False):
                self.episode_low_value_include += 1

            self.query_buffer.append(query_tensor)
            self.response_buffer.append(response_tensor)
            self.reward_buffer.append(reward_tensor)

            stats = self._flush_rollout_if_ready()

            if stats is None:
                loss_str = "buffering"
            else:
                loss_val = stats.get("ppo/loss/total", None)
                if loss_val is None:
                    loss_str = "N/A"
                else:
                    try:
                        if torch.is_tensor(loss_val):
                            loss_val = loss_val.item()
                        loss_str = f"{float(loss_val):.4f}"
                    except Exception:
                        loss_str = str(loss_val)

            print(
                f"Step {step+1:03d} | Action: {'Include(1)' if action==1 else 'Exclude(0)'} "
                f"| Invalid: {invalid_action} | Reward: {reward:+.2f} | Loss: {loss_str} "
                f"| Phase: {self.curriculum.current_phase}"
            )

            if (step + 1) % 10 == 0:
                print(
                    f"[DEBUG:action_dist] include={self.action_counter.get(1, 0)} | "
                    f"exclude={self.action_counter.get(0, 0)} | "
                    f"invalid={self.action_counter.get('invalid', 0)}"
                )

            if is_done or truncated:
                episode_stats = {
                    "is_payload_success": info.get("is_payload_success", False),
                    "payload_score": info.get("payload_score", 0.0)
                }

                print(
                    f"[EPISODE] reward={self.episode_reward:+.2f} | "
                    f"include={self.episode_include} | "
                    f"exclude={self.episode_exclude} | "
                    f"invalid={self.episode_invalid} | "
                    f"selected_tokens={self.episode_selected_tokens} | "
                    f"low_value_include={self.episode_low_value_include} | "
                    f"payload_score={info.get('payload_score')}"
                )

                self.curriculum.record_episode(episode_stats)

                self.episode_reward = 0.0
                self.episode_include = 0
                self.episode_exclude = 0
                self.episode_invalid = 0
                self.episode_selected_tokens = 0
                self.episode_low_value_include = 0

                obs_text, info = self.env.reset()
            else:
                obs_text = next_obs_text

        if len(self.query_buffer) >= self.rollout_batch_size:
            print("[*] 训练结束前，正在刷新最后一个 rollout batch...")
            self._flush_rollout_if_ready()
        else:
            if len(self.query_buffer) > 0:
                print(f"[*] 剩余 {len(self.query_buffer)} 条 rollout 未达到 batch_size，跳过最终 PPO 更新。")

        output_dir = self.default_config["paths"].get(
            "final_lora_dir",
            "./output/vulnseer_selector_lora_final",
        )
        os.makedirs(output_dir, exist_ok=True)
        self.model.pretrained_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"\n[+] 🎉 训练完成！LoRA 权重已保存至 {output_dir}")
