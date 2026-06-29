import os
import json
import argparse
import random


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/valid/test")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于 1.0")

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("输入数据集必须是 list[dict]")

    random.seed(args.seed)
    random.shuffle(dataset)

    n = len(dataset)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    n_test = n - n_train - n_valid

    train_data = dataset[:n_train]
    valid_data = dataset[n_train:n_train + n_valid]
    test_data = dataset[n_train + n_valid:]

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

    print(f"[+] train: {len(train_data)} -> {train_path}")
    print(f"[+] valid: {len(valid_data)} -> {valid_path}")
    print(f"[+] test : {len(test_data)} -> {test_path}")


if __name__ == "__main__":
    main()

# python scripts/split_dataset.py \
#   --input data/dataset_with_gt.json \
#   --output_dir data \
#   --seed 42