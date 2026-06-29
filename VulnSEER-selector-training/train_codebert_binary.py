import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: line {line_no}: {e}")
    return rows


def build_input_text(example: Dict[str, Any]) -> str:
    current_method_signature = example.get("current_method_signature", "")
    current_method_code = example.get("current_method_code", "")
    layer_goal = example.get("layer_goal", {})
    candidate_source_type = example.get("candidate_source_type", "")
    candidate_signature = example.get("candidate_signature", "")
    candidate_content = example.get("candidate_content", "")

    layer_goal_text = json.dumps(layer_goal, ensure_ascii=False)

    text = (
        f"Current Method:\n{current_method_signature}\n\n"
        f"Current Code:\n{current_method_code}\n\n"
        f"Layer Goal:\n{layer_goal_text}\n\n"
        f"Candidate Type:\n{candidate_source_type}\n\n"
        f"Candidate Signature:\n{candidate_signature}\n\n"
        f"Candidate Content:\n{candidate_content}"
    )
    return text


def prepare_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    processed = []
    for ex in rows:
        if "label" not in ex:
            continue

        try:
            label = int(ex["label"])
        except Exception:
            continue

        if label not in (0, 1):
            continue

        processed.append({
            "text": build_input_text(ex),
            "label": label,
            "sample_id": ex.get("sample_id", ""),
            "project_id": ex.get("project_id", ""),
            "candidate_item_id": ex.get("candidate_item_id", ""),
        })

    if not processed:
        raise ValueError("没有可用训练样本，请检查 JSONL 是否包含 text 所需字段和合法 label")
    return Dataset.from_list(processed)


def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_env_info():
    print("=" * 80)
    print("[*] 运行环境信息")
    print(f"[*] torch version          = {torch.__version__}")
    print(f"[*] cuda available         = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] cuda device count      = {torch.cuda.device_count()}")
        print(f"[*] current cuda device    = {torch.cuda.current_device()}")
        print(f"[*] cuda device name       = {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"[*] TOKENIZERS_PARALLELISM = {os.environ.get('TOKENIZERS_PARALLELISM', 'unset')}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="训练集 JSONL")
    parser.add_argument("--val_file", required=True, help="验证集 JSONL")
    parser.add_argument("--model_name", default="microsoft/codebert-base", help="预训练模型名或本地目录")
    parser.add_argument("--output_dir", default="./output/codebert_binary")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # 为了规避环境层不稳定，显式暴露这些参数
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--disable_pin_memory", action="store_true",
                        help="关闭 dataloader pin_memory，建议在 segmentation fault 时开启")
    parser.add_argument("--fp16", action="store_true", help="启用 fp16；默认关闭")
    parser.add_argument("--bf16", action="store_true", help="启用 bf16；默认关闭")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--debug_one_batch", action="store_true",
                        help="只跑一个训练 batch 和一个验证 batch，用于定位是否在训练起步阶段崩溃")
    args = parser.parse_args()

    # 降低 tokenizer 并行带来的不稳定性
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 某些环境下有助于暴露真实报错位置；不会修复 segfault，但便于定位
    if "CUDA_LAUNCH_BLOCKING" not in os.environ:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    set_seed(args.seed)
    print_env_info()

    print("[*] 读取数据集...")
    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)

    train_dataset = prepare_dataset(train_rows)
    val_dataset = prepare_dataset(val_rows)

    print(f"[*] train size = {len(train_dataset)}")
    print(f"[*] val size   = {len(val_dataset)}")

    if args.debug_one_batch:
        print("[*] debug_one_batch=True，仅保留极小数据用于排查")
        train_dataset = train_dataset.select(range(min(8, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(8, len(val_dataset))))
        print(f"[*] debug train size = {len(train_dataset)}")
        print(f"[*] debug val size   = {len(val_dataset)}")

    print("[*] 加载 tokenizer 和 model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    # 明确放到 GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] using device = {device}")
    model.to(device)

    print("[*] 编码数据...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True
    )

    keep_columns = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in train_dataset.column_names:
        keep_columns.append("token_type_ids")

    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in keep_columns]
    )
    val_dataset = val_dataset.remove_columns(
        [c for c in val_dataset.column_names if c not in keep_columns]
    )

    # 显式转成 torch tensor
    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    pin_memory = not args.disable_pin_memory

    print("[*] TrainingArguments 配置如下：")
    print(f"    train_batch_size          = {args.train_batch_size}")
    print(f"    eval_batch_size           = {args.eval_batch_size}")
    print(f"    dataloader_num_workers    = {args.dataloader_num_workers}")
    print(f"    dataloader_pin_memory     = {pin_memory}")
    print(f"    fp16                      = {args.fp16}")
    print(f"    bf16                      = {args.bf16}")
    print(f"    gradient_accumulation     = {args.gradient_accumulation_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        report_to="none",

        # 关键稳定性参数
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_persistent_workers=False,

        # 先默认关闭混合精度，避免底层兼容问题
        fp16=args.fp16,
        bf16=args.bf16,

        # 其他稳妥项
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[*] 开始训练前，先取一个 batch 做冒烟测试...")
    train_loader = trainer.get_train_dataloader()
    first_batch = next(iter(train_loader))
    for k, v in first_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    print("[*] 开始训练...")
    train_result = trainer.train()

    print("[*] 训练完成，开始验证...")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("[*] 保存最佳模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练状态和指标
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[DONE] 模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()