# envs/reward_shaper.py
from typing import Optional
import math

class RewardShaper:
    """
    面向“候选上下文项纳入/不纳入”决策的奖励塑形器
    当前版本核心思想：
    1. 弱化 step reward，避免 PPO 学成局部投机
    2. 强化 final reward，让最终 payload 质量主导优化方向
    3. 长度惩罚改为次线性，避免关键长上下文被误伤
    """
    def __init__(self, config, tokenizer: Optional[object] = None):
        reward_cfg = config.get("reward", {})
        env_cfg = config.get("env_baseline", {})

        self.lambda_cost = reward_cfg.get("lambda_cost", 0.06)
        self.r_slice_success = reward_cfg.get("r_slice_success", 0.08)
        self.r_context_fallback = reward_cfg.get("r_context_fallback", -0.12)
        self.r_progress_weight = reward_cfg.get("r_progress_weight", 0.0)
        self.r_exclude_small_bonus = reward_cfg.get("r_exclude_small_bonus", 0.01)

        self.low_value_penalty = reward_cfg.get("low_value_penalty", 0.25)
        self.high_value_prior = reward_cfg.get("high_value_prior", 0.08)
        self.redundant_include_penalty = reward_cfg.get("redundant_include_penalty", 0.03)
        self.max_cost_penalty = reward_cfg.get("max_cost_penalty", 0.35)

        self.max_success_reward = env_cfg.get("max_success_reward", 8.0)
        self.fail_penalty = env_cfg.get("fail_penalty", -3.0)

        self.tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _cost_penalty(self, token_count: int) -> float:
        # 次线性长度惩罚，避免 token 稍长的关键上下文被过度惩罚
        penalty = self.lambda_cost * math.log1p(token_count)
        return min(penalty, self.max_cost_penalty)

    def _is_low_value_text(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        low_value_markers = [
            "<empty>",
            "fallback",
            "assumed to",
            "generic context",
            "method not found",
            "timeout",
            "no direct field usage found",
            "external library method implementation not available"
        ]
        return any(m in lowered for m in low_value_markers)

    def calculate_step_reward(
        self,
        action: int,
        selected_text: str,
        selection_mode: str,
        progress_delta: int,
        candidate_item: dict = None,
        already_selected_count: int = 0
    ) -> float:
        reward = 0.0
        source_type = (candidate_item or {}).get("source_type", "unknown")

        # 1) exclude：只给非常轻微的预算奖励，防止策略学成“全拒绝”
        if action == 0:
            reward += self.r_exclude_small_bonus
            return reward

        # 2) include：根据来源质量给轻微奖励/惩罚
        if selection_mode == "slice":
            reward += self.r_slice_success
        elif selection_mode == "fallback":
            reward += self.r_context_fallback

        # 3) 长度惩罚：次线性，不要惩罚过重
        token_count = self._count_tokens(selected_text)
        reward -= self._cost_penalty(token_count)

        # 4) 低价值文本惩罚
        if self._is_low_value_text(selected_text):
            reward -= self.low_value_penalty

        # 5) 高价值候选先验：字段/常量/构造函数通常更有用
        if source_type in {"field", "constant", "constructor"}:
            reward += self.high_value_prior

        # 6) 已经选太多，再 include 时逐步加大代价
        if already_selected_count >= 2:
            reward -= self.redundant_include_penalty * (already_selected_count - 1)

        # 7) progress_delta 当前先弱化
        # 注意：当前 solver 的 progress attribution 仍然不够精确，所以这里只保留接口，
        # 默认建议在 config 中设为 0.0，避免错误 credit 误导 PPO。
        if progress_delta > 0:
            reward += self.r_progress_weight * min(progress_delta, 2)

        return reward

    def calculate_final_reward(self, similarity_score: float) -> float:
        # 终局奖励做成主导型
        if similarity_score < 0.30:
            return self.fail_penalty
        elif similarity_score < 0.60:
            return 0.0
        elif similarity_score < 0.80:
            return self.max_success_reward * 0.6
        else:
            return self.max_success_reward * similarity_score