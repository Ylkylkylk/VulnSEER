# trainer/curriculum_manager.py
import yaml
from collections import deque
import numpy as np

class CurriculumManager:
    def __init__(self, config_path: str, env):
        with open(config_path, "r", encoding="utf-8") as f:
            self.curriculum_configs = yaml.safe_load(f)["phases"]

        self.env = env
        self.current_phase = 1
        self.success_history = deque(maxlen=100)
        self._apply_phase_config()

    def _apply_phase_config(self):
        phase_key = f"phase_{self.current_phase}"
        if phase_key not in self.curriculum_configs:
            return

        cfg = self.curriculum_configs[phase_key]
        print(f"\n[🚀 Curriculum Learning] 切换至 {phase_key.upper()}: {cfg['description']}")

        self.env.max_steps_per_episode = cfg["max_steps"]
        self.env.mask_action_0 = cfg["mask_A0"]

        self.env.reward_shaper.lambda_cost = cfg["lambda_cost"]
        self.env.reward_shaper.r_progress_weight = cfg["r_progress_weight"]

        self.env.reward_shaper.r_exclude_small_bonus = cfg.get(
            "r_exclude_small_bonus",
            self.env.reward_shaper.r_exclude_small_bonus
        )
        self.env.reward_shaper.high_value_prior = cfg.get(
            "high_value_prior",
            self.env.reward_shaper.high_value_prior
        )
        self.env.reward_shaper.redundant_include_penalty = cfg.get(
            "redundant_include_penalty",
            self.env.reward_shaper.redundant_include_penalty
        )

        self.threshold = cfg["success_threshold"]

    def record_episode(self, episode_stats: dict):
        success = 1.0 if episode_stats.get("is_payload_success", False) else 0.0
        self.success_history.append(success)

        if len(self.success_history) >= 50 and self.threshold is not None:
            win_rate = np.mean(self.success_history)
            if win_rate >= self.threshold:
                print(f"[*] 当前胜率 {win_rate:.2%} 达到阈值 {self.threshold:.2%}")
                self.current_phase += 1
                self.success_history.clear()
                self._apply_phase_config()