# VulnSEER

VulnSEER is an LLM-guided exploit state re-expression with exploit-relevant context selection framework for client-level exploit payload synthesis in Java projects with vulnerable third-party dependencies.

## Repository Structure

```text
VulnSEER/
├── VulnSEER-main-pipeline/       # CPG processing, call-chain reasoning, and payload generation
├── VulnSEER-selector-training/   # RL selector training and ablation selector training scripts
├── VulnSEER-validation/          # Java replay-based validation engine and evaluation clients
├── requirements.txt              # Python dependencies
├── requirement.txt               # Compatibility alias for requirements.txt
└── README.md
```

Large generated artifacts, model checkpoints, logs, CPG binaries, and validation reports should be kept outside version control.

## Requirements

- Python 3.10 or 3.11
- JDK 17
- Maven 3.8+
- Joern CLI with `joern` and `joern-parse`
- CUDA GPU for RL/LoRA selector training
- An OpenAI-compatible API key for LLM inference

## Environment Setup

```bash
# Create an isolated Python environment for the Python pipeline.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Point VulnSEER to JDK 17 and Joern.
export JAVA_HOME=/path/to/jdk-17
export PATH="$JAVA_HOME/bin:$PATH"
export JOERN_HOME=/path/to/joern-cli
export PATH="$JOERN_HOME:$PATH"

# Configure the LLM endpoint. Do not commit real keys.
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

## Step 1: Generate CPGs

VulnSEER uses Joern CPGs as the static code representation. For a single Java client project, run:

```bash
# Parse one Java client project into a Joern CPG.
# Replace the two paths with your local project path and desired CPG file.
joern-parse /path/to/client-project \
  --language javasrc \
  --output /path/to/project-cpg.bin \
  --frontend-args --enable-file-content
```

For the bundled evaluation clients, VulnSEER also provides a batch helper:

```bash
# Generate CPGs for the configured evaluation clients.
cd VulnSEER-main-pipeline
python batch_generate_cpg.py
```

Optional Joern sanity check:

```scala
importCpg("/path/to/project-cpg.bin")
cpg.method.fullNameExact("fully.qualified.Method:returnType(argTypes)").l
```

## Step 2: Extract Call Chains

The call-chain extraction script queries a CPG for methods matching a sink pattern and emits normalized call-chain records consumed by the main pipeline.

```bash
# Query one CPG for call chains that reach methods matching the sink pattern.
cd VulnSEER-main-pipeline
VULNSEER_CPG_FILE=/path/to/project-cpg.bin \
VULNSEER_SINK="sinkNameOrRegex" \
python joern.py
```

For full reproduction, prepare the project/CVE mapping, CPGs, call-chain records, client projects, and sink-level exploit sketches in the artifact data format shipped with this repository.

## Step 3: Train the RL Selector

The main experiments use an RL-trained context selector. The selector learns whether a candidate code context item should be included during exploit-state resolution.

```bash
# Train the RL selector used by the main pipeline.
cd VulnSEER-selector-training
python main_train.py \
  --config configs/default_config.yaml \
  --curriculum configs/curriculum_config.yaml \
  --dataset data/dataset_with_gt.json \
  --total_timesteps 1000
```

The same directory also includes CodeBERT and Qwen-LoRA selector scripts for ablation studies. Those selectors are optional and are not required for the primary VulnSEER workflow.

## Step 4: Run the Main Pipeline

The main pipeline takes prepared CPG/call-chain inputs, uses the RL selector to choose useful context, asks the solver LLM to resolve exploit states, and produces VulnSEER-format payload JSON objects.

```bash
# Run VulnSEER with the trained RL selector.
# Replace model paths with your local base model and selector checkpoint.
cd VulnSEER-main-pipeline
python run_multi_chain_pipeline.py \
  --policy rl \
  --base_model /path/to/Qwen3-8B \
  --lora_path /path/to/vulnseer-rl-selector \
  --solver_models gpt-5.2
```

Useful options:

- `--solver_models`: LLM used for exploit-goal inference and payload resolution.
- `--base_model`: base model used by the RL selector.
- `--lora_path`: trained RL selector checkpoint.
- `--save_trace`: keep per-layer context-selection traces for debugging.

Optional selector policies for ablation are available through `--policy codebert`, `--policy qwen_lora`, `--policy gpt`, `--policy always_include`, `--policy always_exclude`, and `--policy random`. Use `python run_multi_chain_pipeline.py --help` for the required arguments of each optional policy.

## Step 5: Validate Generated Payloads

The validation engine replays generated VulnSEER payloads against Java client projects and reports whether the vulnerable behavior is reached and triggered.

```bash
# Build the Java validation engine.
cd VulnSEER-validation
mvn clean package
```

```bash
# Validate one generated VulnSEER payload.
# Replace the placeholders with the client project, payload JSON, and validation report directory.
java -jar -noverify vulnseer-engine/target/vulnseer-engine-1.0.jar \
  -p /path/to/client-project \
  -input /path/to/generated-payload.json \
  -output /path/to/validation-report
```

For batch validation over selected bundled clients:

```bash
# Run the bundled parallel validator configuration.
cd VulnSEER-validation
bash run_selected_clients_parallel.sh
```

## Final Artifact

A successful reproduction produces a VulnSEER validation report for each generated payload. The report records the call chain, entry method, sink method, generated payload, replay status, and whether the vulnerability-triggering condition is observed.

## Reproducibility Notes

- Joern parsing is sensitive to Java versions and project dependencies; use JDK 17 unless a client project requires otherwise.
- LLM outputs may vary across providers and model versions. Record the solver model, API endpoint, and selector checkpoint used in each experiment.
- Keep API keys, local absolute paths, model checkpoints, CPG binaries, and generated reports out of Git.
- The validation engine assumes Maven-compatible Java client projects and VulnSEER-format payload JSON.

## Citation

If you use this artifact, please cite the corresponding VulnSEER paper.
