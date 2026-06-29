# VulnSEER

VulnSEER is a reproducibility artifact for LLM-guided vulnerability exploitability analysis. Given a Java client project, VulnSEER builds code-property graph context, constructs entry-to-sink execution plans, resolves exploit states with an LLM, optionally selects useful context with a trained selector, and validates generated payloads with the validation engine.

The repository is organized to mirror the paper workflow:

1. Build CPGs and call chains.
2. Infer sink requirements and layer-wise exploit goals.
3. Train or choose a context selector.
4. Run the main payload-generation pipeline.
5. Dynamically validate generated payloads.

## Repository Layout

```text
VulnSEER/
├── VulnSEER-main-pipeline/       # CPG/call-chain processing and payload generation
├── VulnSEER-selector-training/   # RL, CodeBERT, and Qwen-LoRA selector training
├── VulnSEER-validation/          # Java validation engine and client applications
├── requirements.txt              # Python dependencies
└── README.md
```

Generated artifacts such as CPG binaries, model checkpoints, logs, and pipeline outputs are not intended to be committed.

## Environment

We recommend Linux with CUDA for selector training. Payload generation and validation can run on CPU, but selector training requires a GPU for practical runtime.

Required external tools:

- Python 3.10 or 3.11
- JDK 17
- Maven 3.8+
- Joern CLI with `joern` and `joern-parse`
- CUDA-compatible GPU for RL/LoRA selector training

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set environment variables:

```bash
export JAVA_HOME=/path/to/jdk-17
export PATH="$JAVA_HOME/bin:$PATH"

export JOERN_HOME=/path/to/joern-cli
export PATH="$JOERN_HOME:$PATH"

export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

Do not commit API keys or local machine paths.

## Step 1: Generate CPGs

VulnSEER uses Joern CPGs as the primary code representation. Put Java client projects under:

```text
VulnSEER-main-pipeline/client-apps/
```

Generate one CPG manually:

```bash
cd VulnSEER-main-pipeline
mkdir -p cpg-file

joern-parse client-apps/adyen-api \
  --language javasrc \
  --output cpg-file/adyen-api-cpg.bin \
  --frontend-args --enable-file-content
```

Batch-generate CPGs for the configured client projects:

```bash
cd VulnSEER-main-pipeline
python batch_generate_cpg.py
```

Optional CPG sanity check in Joern:

```scala
importCpg("cpg-file/adyen-api-cpg.bin")
cpg.method.fullNameExact("fully.qualified.Method:ret(args)").l
```

## Step 2: Extract Call Chains

Use `joern.py` to query call chains from a CPG to a sink-pattern:

```bash
cd VulnSEER-main-pipeline

VULNSEER_CPG_FILE=cpg-file/gerenciador-viagens-cpg.bin \
VULNSEER_SINK=matches \
VULNSEER_CALLCHAINS_OUTPUT=call-chain-output/gerenciador-viagens-callchains.json \
python joern.py
```

For full reproduction, prepare:

```text
VulnSEER-main-pipeline/cpg-file/*.bin
VulnSEER-main-pipeline/call-chain-output/*-callchains.json
VulnSEER-main-pipeline/file-cve-1.json
VulnSEER-main-pipeline/exploit-sketches/
```

`exploit-sketches/` contains sink-level exploit sketches used to infer generalized sink requirements.

## Step 3: Train Selectors

Selector training code is in:

```bash
cd VulnSEER-selector-training
```

### RL Selector

```bash
VULNSEER_SELECTOR_BASE_MODEL=/path/to/Qwen3-8B \
OPENAI_API_KEY="YOUR_API_KEY" \
python main_train.py \
  --config configs/default_config.yaml \
  --curriculum configs/curriculum_config.yaml \
  --dataset data/dataset_with_gt.json \
  --total_timesteps 1000
```

The final LoRA checkpoint is written to:

```text
VulnSEER-selector-training/output/vulnseer_selector_lora_final/
```

### CodeBERT Selector

```bash
python train_codebert_binary.py \
  --train_file data/codebert_binary_train.jsonl \
  --val_file data/codebert_binary_val.jsonl \
  --model_name /path/to/codebert-base \
  --output_dir output/codebert_binary \
  --max_length 512 \
  --learning_rate 2e-5 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --num_train_epochs 5 \
  --disable_pin_memory \
  --dataloader_num_workers 0
```

### Qwen-LoRA Binary Selector

```bash
python train_qwen3_lora_cls.py \
  --train_file data/codebert_binary_train.jsonl \
  --val_file data/codebert_binary_val.jsonl \
  --model_name /path/to/Qwen3-8B \
  --output_dir output/qwen3_lora_binary \
  --max_length 1024 \
  --learning_rate 1e-4 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --bf16
```

## Step 4: Run the Main Pipeline

Main pipeline entry:

```bash
cd VulnSEER-main-pipeline
```

The default run uses `file-cve-1.json`, `cpg-file/`, `call-chain-output/`, `exploit-sketches/`, and `client-apps/` under `VulnSEER-main-pipeline`.

### Always-Include Baseline

```bash
OPENAI_API_KEY="YOUR_API_KEY" \
OPENAI_BASE_URL="https://openrouter.ai/api/v1" \
python run_multi_chain_pipeline.py \
  --policy always_include \
  --solver_models gpt-5.2
```

### RL Selector

```bash
OPENAI_API_KEY="YOUR_API_KEY" \
python run_multi_chain_pipeline.py \
  --policy rl \
  --base_model /path/to/Qwen3-8B \
  --lora_path ../VulnSEER-selector-training/output/vulnseer_selector_lora_final \
  --solver_models gpt-5.2
```

### CodeBERT Selector

```bash
OPENAI_API_KEY="YOUR_API_KEY" \
python run_multi_chain_pipeline.py \
  --policy codebert \
  --codebert_model_path ../VulnSEER-selector-training/output/codebert_binary \
  --codebert_threshold 0.7 \
  --solver_models gpt-5.2
```

### LLM Selector

```bash
OPENAI_API_KEY="YOUR_API_KEY" \
python run_multi_chain_pipeline.py \
  --policy gpt \
  --gpt_model_name gpt-5.2 \
  --gpt_base_url "$OPENAI_BASE_URL" \
  --solver_models gpt-5.2
```

### Qwen-LoRA Selector

```bash
OPENAI_API_KEY="YOUR_API_KEY" \
python run_multi_chain_pipeline.py \
  --policy qwen_lora \
  --base_model /path/to/Qwen3-8B \
  --lora_path ../VulnSEER-selector-training/output/qwen3_lora_binary \
  --qwen_lora_threshold 0.7 \
  --qwen_lora_max_length 1024 \
  --solver_models gpt-5.2
```

Outputs are written to model-specific `multi-chain-output-*` directories under `VulnSEER-main-pipeline`.

## Step 5: Validate Generated Payloads

Build the Java validation engine:

```bash
cd VulnSEER-validation
mvn clean package
```

Run validation for one generated input file:

```bash
java -jar -noverify vulnseer-engine/target/vulnseer-engine-1.0.jar \
  -p client-apps/flow \
  -input input/flow/gpt5.2-rl/input.json \
  -output output/flow-gpt5.2-rl
```

For batch validation, use:

```bash
cd VulnSEER-validation
bash run_selected_clients_parallel.sh
```

## Expected Artifacts

A successful run produces:

- CPG files: `VulnSEER-main-pipeline/cpg-file/*.bin`
- Call chains: `VulnSEER-main-pipeline/call-chain-output/*.json`
- Execution plans and payloads: `VulnSEER-main-pipeline/multi-chain-output-*`
- Selector checkpoints: `VulnSEER-selector-training/output/*`
- Validation outputs: `VulnSEER-validation/output*`

## Reproducibility Notes

- Joern parsing is sensitive to Java version and project dependencies. Use JDK 17 unless the target client project requires otherwise.
- LLM results may vary across model providers and model versions. Record `OPENAI_BASE_URL`, model names, and decoding settings for experiments.
- Large binary artifacts and downloaded model checkpoints should be stored outside Git or released separately.
- The validation engine assumes Maven-compatible Java client projects and generated input JSON files in VulnSEER format.

## Citation

If you use this artifact, please cite the corresponding VulnSEER paper.
