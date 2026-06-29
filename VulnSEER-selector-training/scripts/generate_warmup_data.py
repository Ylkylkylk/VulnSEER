# scripts/generate_warmup_data.py
import json
import os

def generate_phase1_warmup_data(raw_callchains_json, output_path):
    """
    筛选跳数 <= 2 且方法体内代码行数较少、无大量日志干扰的干净调用链，供 Phase 1 热身使用。
    """
    print("[*] 正在筛选 Phase 1 热身数据...")
    with open(raw_callchains_json, "r") as f:
        chains = json.load(f)
        
    warmup_chains = []
    for chain in chains:
        # 1. 筛选短链 (1-Hop 或 2-Hop)
        if len(chain) <= 3: 
            # 2. 可以在此处加入对 Joern CPG 结果的预检，过滤掉含有大量 org.slf4j 的方法
            # 这里简化为长度判断
            warmup_chains.append(chain)
            
    with open(output_path, "w") as f:
        json.dump(warmup_chains, f, indent=4)
    print(f"[+] 成功筛选 {len(warmup_chains)} 条 Phase 1 数据，保存至 {output_path}")

if __name__ == "__main__":
    generate_phase1_warmup_data(
        raw_callchains_json="../call-chain-output/gerenciador-viagens-callchains.json",
        output_path="../data/phase1_warmup.json"
    )