# agent/rl_buffer.py
import torch
import numpy as np
from typing import List, Dict

class VulnRolloutBuffer:
    """
    为漏洞推理环境定制的 PPO 经验回放池。
    支持可变长度的文本状态 (Prompt) 存储和 GAE (Generalized Advantage Estimation) 计算。
    """
    def __init__(self, gamma: float = 0.99, lam: float = 0.95, device: str = "cpu"):
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.reset()

    def reset(self):
        """清空当前 Buffer，通常在每次 PPO 更新后调用"""
        self.states: List[str] = []           # 文本 Prompt
        self.actions: List[int] = []          # 离散动作 (0 或 1)
        self.rewards: List[float] = []        # 奖励
        self.values: List[float] = []         # Critic 预测的状态价值
        self.log_probs: List[float] = []      # 动作的对数概率
        self.dones: List[bool] = []           # 是否结束

    def add(self, state: str, action: int, reward: float, value: float, log_prob: float, done: bool):
        """向 Buffer 中添加一步交互经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float, done: bool) -> Dict[str, torch.Tensor]:
        """
        基于收集到的轨迹，计算 GAE 和目标 Returns。
        这能有效平衡方差和偏差，尤其是在我们的长链条深度推理 (Phase 3) 中。
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [done])

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0

        # 从后向前递归计算 GAE
        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - float(dones[step + 1])
            delta = rewards[step] + self.gamma * values[step + 1] * next_non_terminal - values[step]
            advantages[step] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam

        returns = advantages + values[:-1]

        # 转换为 Tensor
        return {
            "advantages": torch.tensor(advantages, dtype=torch.float32).to(self.device),
            "returns": torch.tensor(returns, dtype=torch.float32).to(self.device),
            "states": self.states,
            "actions": torch.tensor(self.actions, dtype=torch.long).to(self.device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)
        }