import os
import json
import argparse
import random


def main():
    parser = argparse.ArgumentParser(description="Randomly split dataset into 42/8/7")
    parser.add_argument("--input", type=str, required=True, help="原始数据集路径")
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("输入数据集必须是 list")

    if len(dataset) != 57:
        raise ValueError(f"当前数据集条数为 {len(dataset)}，不是 57，无法按 42/8/7 精确划分。")

    random.seed(args.seed)
    shuffled = dataset[:]
    random.shuffle(shuffled)

    train_data = shuffled[:42]
    valid_data = shuffled[42:50]
    test_data = shuffled[50:57]

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "dataset_train.json")
    valid_path = os.path.join(args.output_dir, "dataset_valid.json")
    test_path = os.path.join(args.output_dir, "dataset_test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(valid_path, "w", encoding="utf-8") as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"[+] train = {len(train_data)} -> {train_path}")
    print(f"[+] valid = {len(valid_data)} -> {valid_path}")
    print(f"[+] test  = {len(test_data)} -> {test_path}")


if __name__ == "__main__":
    main()