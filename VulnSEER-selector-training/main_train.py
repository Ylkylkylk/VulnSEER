# main_train.py
import argparse
import os
import json

def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset(dataset_path: str) -> list:
    print(f"[*] 正在加载训练数据集: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="RL-based Context Selection for Vulnerability Reasoning")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--curriculum", type=str, default="configs/curriculum_config.yaml")
    parser.add_argument("--dataset", type=str, default="data/dataset_with_gt.json")
    parser.add_argument("--total_timesteps", type=int, default=1000)
    parser.add_argument("--do_warmstart", action="store_true")
    parser.add_argument("--warmstart_epochs", type=int, default=1)
    args = parser.parse_args()

    from tools.joern_slicer import JoernSlicer
    from envs.vuln_reasoning_env import VulnReasoningEnv
    from agent.policy_model import PolicyModel
    from trainer.skyrl_trainer import VulnRLTrainer
    from bottom_up_inferencer import LayerWiseExploitGoalInferencer, ExploitStateResolver

    print("[*] 正在加载配置文件...")
    config = load_config(args.config)

    training_dataset = load_dataset(args.dataset)
    split_idx = max(1, int(len(training_dataset) * 0.8))
    train_dataset = training_dataset[:split_idx]
    val_dataset = training_dataset[split_idx:] if split_idx < len(training_dataset) else training_dataset[:min(10, len(training_dataset))]

    print("[*] 正在初始化 Joern 引擎与底层推理框架...")
    cpg_path = config["env_baseline"]["cpg_path"]
    joern_bin = config["env_baseline"]["joern_bin"]

    inferencer = LayerWiseExploitGoalInferencer(
        cpg_path=cpg_path,
        joern_bin=joern_bin,
        model=config["solver_llm"]["model_name"],
        debug_prompt_dir=config["paths"]["debug_prompt_dir"]
    )
    # inferencer = LayerWiseExploitGoalInferencer(
    #     cpg_path=cpg_path,
    #     joern_bin=joern_bin,
    #     model=config["solver_llm"]["model_name"],
    #     api_key_env=config["solver_llm"].get("api_key_env", "OPENAI_API_KEY"),
    #     base_url=config["solver_llm"].get("base_url"),
    #     debug_prompt_dir=config["paths"]["debug_prompt_dir"]
    # )

    exploit_state_resolver = ExploitStateResolver(inferencer)

    print("[*] 正在构建策略网络 (Actor-Critic with LoRA)...")
    policy_model = PolicyModel(config=config)
    _, tokenizer = policy_model.get_model_and_tokenizer()

    if args.do_warmstart:
        from trainer.warmstart_trainer import WarmStartTrainer
        warm_trainer = WarmStartTrainer(
            policy_model=policy_model,
            inferencer=inferencer,
            dataset=training_dataset,
            tokenizer=tokenizer,
            config=config
        )
        warm_trainer.train(epochs=args.warmstart_epochs)

    joern_slicer = JoernSlicer(
        cpg_path=cpg_path,
        joern_bin=joern_bin,
        temp_dir=config["paths"]["temp_dir"]
    )

    print("[*] 正在构建 VulnReasoningEnv (MDP 环境)...")
    env = VulnReasoningEnv(
        config=config,
        exploit_state_resolver=exploit_state_resolver,
        joern_slicer=joern_slicer,
        dataset=train_dataset,
        tokenizer=tokenizer,
        use_llm_judge=False
    )

    print("[*] 正在装配 Trainer 与课程学习调度器...")
    trainer = VulnRLTrainer(
        env=env,
        policy_model=policy_model,
        default_config=config,
        curriculum_config_path=args.curriculum,
        val_dataset=val_dataset,
        inferencer=inferencer,
        joern_slicer=joern_slicer
    )

    print("\n" + "="*50)
    print(f"🚀 开始强化学习微调 (Total Timesteps: {args.total_timesteps})")
    print("="*50 + "\n")

    try:
        trainer.train(total_timesteps=args.total_timesteps)
    except KeyboardInterrupt:
        print("\n[!] 检测到手动中断，正在保存当前 LoRA 权重...")
        save_path = config["paths"]["interrupted_lora_dir"]
        os.makedirs(save_path, exist_ok=True)
        trainer.model.pretrained_model.save_pretrained(save_path)
        trainer.tokenizer.save_pretrained(save_path)
        print(f"[+] 权重已保存至 {save_path}")

if __name__ == "__main__":
    main()
