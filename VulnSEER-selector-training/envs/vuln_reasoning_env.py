# envs/vuln_reasoning_env.py
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Dict, Any, Tuple

from .reward_shaper import RewardShaper
from tools.payload_evaluator import ExploitPayloadEvaluator
from tools.context_fallback import ContextFallbackBuilder
from tools.state_formatter import StateFormatter


class VulnReasoningEnv(gym.Env):
    """
    新版环境：
    决策对象 = helpers / class_context 的候选上下文项
    动作语义：
      0 -> exclude
      1 -> include (先 slice，失败则 fallback)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exploit_state_resolver,
        joern_slicer,
        dataset: list,
        tokenizer=None,
        use_llm_judge=False
    ):
        super().__init__()
        self.config = config
        self.solver = exploit_state_resolver
        self.slicer = joern_slicer
        self.dataset = dataset

        self.reward_shaper = RewardShaper(config, tokenizer=tokenizer)
        self.evaluator = ExploitPayloadEvaluator(use_llm_judge=use_llm_judge)
        self.context_fallback = ContextFallbackBuilder()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Text(max_length=8192)

        self.current_phase = 1
        self.max_steps_per_episode = config.get("env_baseline", {}).get("max_steps", 100) \
            if "env_baseline" in config else config.get("max_steps", 100)
        self.mask_action_0 = False

        self.current_step = 0
        self.current_candidate_item = None
        self.current_method = None
        self.current_layer_goal = None
        self.current_method_code = None
        self.current_selected_context = None
        self.current_gt_payload = None
        self.solver_generator = None
        

    def reset(self, seed=None, options=None) -> Tuple[str, dict]:
        super().reset(seed=seed)

        if not self.dataset:
            raise ValueError("dataset 为空，无法训练。")

        self.current_step = 0
        self.current_candidate_item = None
        self.current_method = None
        self.current_layer_goal = None
        self.current_method_code = None
        self.current_selected_context = None
        self.current_gt_payload = None
        self.solver_generator = None

        max_retry = 5
        for _ in range(max_retry):
            print("[DEBUG] reset() 正在尝试采样一个包含可决策上下文项的样本...")
            sample = random.choice(self.dataset)

            sample_id = sample.get("sample_id", "unknown_sample")
            project_id = sample.get("project_id", "unknown_project")
            current_cpg_path = sample.get("cpg_path", None)

            if not current_cpg_path:
                raise ValueError(f"sample {sample_id} 缺少 cpg_path，无法切换 CPG。")

            # 动态切换当前 sample 对应的 CPG
            self.solver.set_cpg_path(current_cpg_path)
            self.slicer.set_cpg_path(current_cpg_path)

            print(f"[DEBUG] 当前 sample_id={sample_id}")
            print(f"[DEBUG] 当前 project_id={project_id}")
            print(f"[DEBUG] 当前 cpg_path={current_cpg_path}")

            current_execution_plan = sample["execution_plan"]
            self.current_gt_payload = sample["ground_truth_payload"]
            print(f"[DEBUG] 当前 sample_id={sample.get('sample_id', 'unknown')}")

            self.solver_generator = self.solver.resolve_execution_plan(current_execution_plan)
            event = next(self.solver_generator)

            if not event.get("is_done", False):
                self.current_candidate_item = event["candidate_item"]
                self.current_method = event["current_method"]
                self.current_layer_goal = event["constraint"]
                self.current_method_code = event["method_code"]
                self.current_selected_context = event.get("already_selected", {})

                state_text = StateFormatter.build_prompt(
                    layer_goal=self.current_layer_goal,
                    candidate_item=self.current_candidate_item,
                    method_code=self.current_method_code,
                    already_selected=self.current_selected_context,
                    layer_index=event.get("layer_index"),
                    max_steps=self.max_steps_per_episode
                )
                return state_text, {"phase": self.current_phase}

        raise RuntimeError("连续采样后都没有可决策的候选上下文项，当前数据集不适合 RL 训练。")

    def step(self, action: int):
        if self.current_candidate_item is None:
            raise RuntimeError("当前没有可决策的 candidate_item。")

        self.current_step += 1

        if self.mask_action_0 and action == 0:
            action = 1

        selected_text = ""
        selection_mode = "exclude"

        if action == 1:
            # 优先 slice
            sliced_text = self.slicer.slice_context_item(
                current_method=self.current_method,
                item=self.current_candidate_item
            )

            if sliced_text and "not found" not in sliced_text.lower() and "timeout" not in sliced_text.lower():
                selected_text = sliced_text
                selection_mode = "slice"
            else:
                selected_text = self.context_fallback.generate_fallback_for_context_item(self.current_candidate_item)
                selection_mode = "fallback"

        solver_feedback = {
            "action": action,
            "selected_text": selected_text,
            "source_type": self.current_candidate_item.get("source_type", "unknown")
        }

        event = self.solver_generator.send(solver_feedback)

        raw_progress_delta = event.get("progress_delta", 0)

        # 当前版本先不把 raw_progress_delta 直接用于 step reward，
        # 因为它并不总是“当前动作立即造成的真实推进量”
        effective_progress_delta = 0

        already_selected_count = 0
        if self.current_selected_context:
            already_selected_count += len(self.current_selected_context.get("helpers", []) or [])
            already_selected_count += len(self.current_selected_context.get("class_context", []) or [])

        reward = self.reward_shaper.calculate_step_reward(
            action=action,
            selected_text=selected_text,
            selection_mode=selection_mode,
            progress_delta=effective_progress_delta,
            candidate_item=self.current_candidate_item,
            already_selected_count=already_selected_count
        )

        is_done = event.get("is_done", False)
        truncated = self.current_step >= self.max_steps_per_episode
        score = 0.0

        selected_text_token_count = self.reward_shaper._count_tokens(selected_text) if selected_text else 0
        selected_text_is_low_value = self.reward_shaper._is_low_value_text(selected_text) if selected_text else False

        if is_done or truncated:
            final_generated_payload = event.get("final_payload", {})

            if is_done and final_generated_payload:
                print("[*] Episode 结束，正在评估生成的 sink requirement...")
                score = self.evaluator.evaluate(
                    generated_payload=final_generated_payload,
                    ground_truth_payload=self.current_gt_payload
                )
                print(f"[+] 裁判打分结果: {score:.2f}")
            else:
                print("[-] 推导失败或超时截断，得分为 0.0")

            reward += self.reward_shaper.calculate_final_reward(score)

            info = {
                "action_taken": action,
                "selection_mode": selection_mode,
                "current_step": self.current_step,
                "payload_score": score,
                "is_payload_success": score >= 0.6,
                "candidate_signature": self.current_candidate_item.get("signature", ""),
                "candidate_source_type": self.current_candidate_item.get("source_type", "unknown"),
                "progress_delta": raw_progress_delta,
                "selected_text_token_count": selected_text_token_count,
                "selected_text_is_low_value": selected_text_is_low_value
            }
            return "DONE", reward, True, truncated, info

        self.current_candidate_item = event["candidate_item"]
        self.current_method = event["current_method"]
        self.current_layer_goal = event["constraint"]
        self.current_method_code = event["method_code"]
        self.current_selected_context = event.get("already_selected", {})

        next_state_text = StateFormatter.build_prompt(
            layer_goal=self.current_layer_goal,
            candidate_item=self.current_candidate_item,
            method_code=self.current_method_code,
            already_selected=self.current_selected_context,
            layer_index=event.get("layer_index"),
            max_steps=self.max_steps_per_episode
        )

        info = {
            "action_taken": action,
            "selection_mode": selection_mode,
            "current_step": self.current_step,
            "payload_score": None,
            "is_payload_success": False,
            "candidate_signature": self.current_candidate_item.get("signature", ""),
            "candidate_source_type": self.current_candidate_item.get("source_type", "unknown"),
            "progress_delta": raw_progress_delta,
            "selected_text_token_count": selected_text_token_count,
            "selected_text_is_low_value": selected_text_is_low_value
        }
        return next_state_text, reward, False, False, info