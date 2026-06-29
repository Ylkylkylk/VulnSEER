import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Tuple

from bottom_up_inferencer import (
    ExecutionPlanBuilder,
    ExploitStateResolver,
    LayerWiseExploitGoalInferencer,
)
from policy_selectors.context_selectors import build_selector
from tools.joern_slicer import JoernSlicer
from tools.code_mocker import ContextFallbackBuilder
from tools.path_context_indexer import PathContextIndexer


# =========================================================
# Default configuration
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) Batch project/CVE mapping file.
PROJECT_CVE_MAP_JSON = os.path.join(SCRIPT_DIR, "file-cve-1.json")

# 2) Root directories. Shared naming conventions are composed from here.
CPG_ROOT = os.path.join(SCRIPT_DIR, "cpg-file")
CALLCHAINS_ROOT = os.path.join(SCRIPT_DIR, "call-chain-output")
EXPLOIT_SKETCH_ROOT = os.path.join(SCRIPT_DIR, "exploit-sketches")
JOERN_BIN = os.environ.get("JOERN_BIN", "joern")
SOURCE_APPS_ROOT = os.path.join(SCRIPT_DIR, "client-apps")

# 3) Multi-model batch run.
# SOLVER_MODELS = [
#     "gpt-5.2",
#     "gpt-4o",
#     "claude-sonnet-4-6",
#     "deepseek-v3.2",
#     "qwen3.5-plus",
# ]

SOLVER_MODELS = [
    "gpt-5.2",
]

# 4) Map each solver model to its output root.
# By default, outputs are placed under the script directory; per-model overrides can be set here.
MODEL_OUTPUT_ROOT_MAP = {
    "gpt-5.2": os.path.join(SCRIPT_DIR, "multi-chain-output-gpt5.2"),
}

# 5) Selector policy configuration.
# Available policies:
#   rl              : Original CausalLM + LoRA selector that generates 0/1 decisions.
#   base            : Original CausalLM base selector that generates 0/1 decisions.
#   codebert        : CodeBERT binary selector.
#   qwen_lora       : Qwen3-8B + LoRA binary selector.
#   gpt             : GPT API selector
#   always_include  : Always include.
#   always_exclude  : Always exclude.
#   random          : Randomly include or exclude.
POLICY = "always_include"

BASE_MODEL = None
LORA_PATH = None

CODEBERT_MODEL_PATH = None
CODEBERT_THRESHOLD = 0.5

# Qwen3-8B-LoRA binary selector parameters.
QWEN_LORA_THRESHOLD = 0.5
QWEN_LORA_MAX_LENGTH = 1024
QWEN_LORA_LOAD_IN_4BIT = True

GPT_MODEL_NAME = "deepseek-v3.2"
GPT_BASE_URL = None

MAX_STEPS = 100
SAVE_TRACE = False


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    text = text.strip("._")
    if not text:
        text = "chain"
    if len(text) > max_len:
        text = text[:max_len]
    return text


def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def derive_cpg_project_name(project_name: str) -> str:
    """
    Recover the real CPG prefix from a project_name that may include a vulnerability suffix.

    Examples:
      adu-test-2019-10086   -> adu-test
      adu-test-2020-13956   -> adu-test
      adu-test-LANG-1645    -> adu-test
      adyen-api             -> adyen-api
      archivefs             -> archivefs
    """
    parts = project_name.split("-")
    if len(parts) <= 1:
        return project_name

    suffix_start = None

    for i, part in enumerate(parts):
        # Case 1: ...-2019-10086
        if re.fullmatch(r"\d{4}", part):
            suffix_start = i
            break

        # Case 2: ...-LANG-1645 / ...-CVE-2021-1234
        if re.fullmatch(r"[A-Za-z]+", part):
            if i + 1 < len(parts) and re.fullmatch(r"\d+", parts[i + 1]):
                suffix_start = i
                break
            if (
                i + 2 < len(parts)
                and re.fullmatch(r"\d{4}", parts[i + 1])
                and re.fullmatch(r"\d+", parts[i + 2])
            ):
                suffix_start = i
                break

    if suffix_start is None:
        return project_name

    base = "-".join(parts[:suffix_start]).strip("-")
    return base if base else project_name


def load_project_cve_map(project_cve_map_json: str) -> List[Dict[str, str]]:
    """
    Triple mapping format:
    [
      {
        "project": "adu-test-2019-10086",
        "cve": "CVE-2019-10086",
        "cpg_project": "adu-test"
      },
      ...
    ]
    """
    if not os.path.exists(project_cve_map_json):
        raise FileNotFoundError(f"project_cve_map_json not found: {project_cve_map_json}")

    with open(project_cve_map_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("project_cve_map_json must be in list[dict] format")

    result: List[Dict[str, str]] = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not a dict: {item}")

        project_name = str(item.get("project", "")).strip()
        cve_name = str(item.get("cve", "")).strip()
        cpg_project = str(item.get("cpg_project", "")).strip()

        if not project_name:
            raise ValueError(f"Item {idx} is missing project")
        if not cve_name:
            raise ValueError(f"Item {idx} is missing cve")
        if not cpg_project:
            raise ValueError(f"Item {idx} is missing cpg_project")

        result.append({
            "project": project_name,
            "cve": cve_name,
            "cpg_project": cpg_project,
        })

    return result


def load_callchains(callchains_json: str) -> List[Dict[str, Any]]:
    if not os.path.exists(callchains_json):
        raise FileNotFoundError(f"Call-chain file not found: {callchains_json}")

    with open(callchains_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized: List[Dict[str, Any]] = []

    # Form 1: directly [[m1,m2,...], [...]]
    if isinstance(data, list) and data and all(isinstance(x, list) for x in data):
        for idx, chain in enumerate(data):
            normalized.append({
                "chain_id": f"chain_{idx:03d}",
                "chain": chain
            })
        return normalized

    # Form 2: [{"chain": [...]}, ...]
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        for idx, item in enumerate(data):
            chain = item.get("chain") or item.get("call_chain") or item.get("methods")
            if not isinstance(chain, list):
                continue
            normalized.append({
                "chain_id": item.get("chain_id") or item.get("id") or f"chain_{idx:03d}",
                "chain": chain,
                "meta": {k: v for k, v in item.items() if k not in {"chain", "call_chain", "methods"}}
            })
        return normalized

    # Form 3: {"chains": [...]} / {"call_chains": [...]} / {"data": [...]}
    if isinstance(data, dict):
        chains = data.get("chains") or data.get("call_chains") or data.get("data")
        if isinstance(chains, list):
            if chains and all(isinstance(x, list) for x in chains):
                for idx, chain in enumerate(chains):
                    normalized.append({
                        "chain_id": f"chain_{idx:03d}",
                        "chain": chain
                    })
                return normalized

            if chains and all(isinstance(x, dict) for x in chains):
                for idx, item in enumerate(chains):
                    chain = item.get("chain") or item.get("call_chain") or item.get("methods")
                    if not isinstance(chain, list):
                        continue
                    normalized.append({
                        "chain_id": item.get("chain_id") or item.get("id") or f"chain_{idx:03d}",
                        "chain": chain,
                        "meta": {k: v for k, v in item.items() if k not in {"chain", "call_chain", "methods"}}
                    })
                return normalized

    raise ValueError("Unrecognized call-chain JSON format; expected list[list[str]] or JSON objects with a chain field")


def build_file_stem(chain_id: str, chain: List[str]) -> str:
    entry = chain[0] if chain else "entry"
    sink = chain[-1] if chain else "sink"
    entry_name = entry.split(":")[0].split(".")[-1]
    sink_name = sink.split(":")[0].split(".")[-1]
    stem = f"{chain_id}_{entry_name}_to_{sink_name}"
    return sanitize_filename(stem)


def build_solver_output_base_dir(solver_model: str, project_name: str) -> str:
    """
    Example:
      gpt-5.2 + webbit -> ./multi-chain-output-gpt5.2/webbit
    """
    if solver_model in MODEL_OUTPUT_ROOT_MAP:
        return os.path.join(MODEL_OUTPUT_ROOT_MAP[solver_model], project_name)

    safe_model = sanitize_filename(solver_model)
    return os.path.join(SCRIPT_DIR, f"multi-chain-output-{safe_model}", project_name)


def build_policy_output_dir(args) -> str:
    """
    Write different policies to different directories to avoid overwriting results.
    Directory examples:
      /.../multi-chain-output-gpt5.2/webbit/always_include
      /.../multi-chain-output-gpt5.2/webbit/codebert_codebert_binary_thr0_5
      /.../multi-chain-output-gpt5.2/webbit/qwen_lora_Qwen3-8B-2_qwen3_lora_binary_thr0_5_len1024
    """
    policy = args.policy

    if policy == "rl":
        model_name = sanitize_filename(os.path.basename(args.base_model or "base"))
        lora_name = sanitize_filename(os.path.basename(args.lora_path or "lora"))
        policy_tag = f"rl_{model_name}_{lora_name}"

    elif policy == "base":
        model_name = sanitize_filename(os.path.basename(args.base_model or "base"))
        policy_tag = f"base_{model_name}"

    elif policy == "codebert":
        model_name = sanitize_filename(os.path.basename(args.codebert_model_path or "codebert"))
        thr = str(args.codebert_threshold).replace(".", "_")
        policy_tag = f"codebert_{model_name}_thr{thr}"

    elif policy == "qwen_lora":
        model_name = sanitize_filename(os.path.basename(args.base_model or "qwen_base"))
        lora_name = sanitize_filename(os.path.basename(args.lora_path or "qwen_lora"))
        thr = str(args.qwen_lora_threshold).replace(".", "_")
        policy_tag = f"qwen_lora_{model_name}_{lora_name}_thr{thr}_len{args.qwen_lora_max_length}"
        if args.qwen_lora_no_4bit:
            policy_tag += "_no4bit"
        else:
            policy_tag += "_4bit"

    elif policy == "gpt":
        model_name = sanitize_filename(args.gpt_model_name or "gpt")
        policy_tag = f"gpt_{model_name}"

    else:
        policy_tag = sanitize_filename(policy)

    return os.path.join(args.output_base_dir, policy_tag)


def resolve_solver_models(args) -> List[str]:
    if args.solver_models:
        models = [m.strip() for m in args.solver_models if m and m.strip()]
        if models:
            return models
    if SOLVER_MODELS:
        return [m.strip() for m in SOLVER_MODELS if m and m.strip()]
    raise ValueError("No runnable solver models")


def build_project_run_args(
    base_args,
    project_name: str,
    cve_name: str,
    cpg_project: str,
    solver_model: str,
):
    """
    Compose the core paths needed for one project run from the triple mapping.
    """
    run_args = argparse.Namespace(**vars(base_args))

    run_args.project_name = project_name
    run_args.cve_name = cve_name
    run_args.cpg_project = cpg_project
    run_args.solver_model = solver_model

    # Use only cpg_project for the CPG file.
    run_args.cpg_file = os.path.join(base_args.cpg_root, f"{cpg_project}-cpg.bin")

    # Keep using the full project name for call-chain files.
    run_args.callchains_json = os.path.join(
        base_args.callchains_root,
        f"{project_name}-callchains.json"
    )

    run_args.output_base_dir = build_solver_output_base_dir(solver_model, project_name)

    run_args.exploit_sketch_dir = os.path.join(
        base_args.exploit_sketch_root,
        cve_name,
        "src", "main", "java", "com", "vulnseer", "validation"
    )

    run_args.source_root = (
        base_args.source_root
        if getattr(base_args, "source_root", None)
        else os.path.join(SOURCE_APPS_ROOT, cpg_project)
    )

    run_args.joern_bin = base_args.joern_bin
    return run_args


def validate_required_inputs(run_args):
    missing = []

    if not os.path.exists(run_args.cpg_file):
        missing.append(f"CPG_FILE not found: {run_args.cpg_file}")

    if not os.path.exists(run_args.callchains_json):
        missing.append(f"CALLCHAINS_JSON not found: {run_args.callchains_json}")

    if not os.path.exists(run_args.exploit_sketch_dir):
        missing.append(f"EXPLOIT_SKETCH_DIR not found: {run_args.exploit_sketch_dir}")

    if not os.path.exists(run_args.joern_bin):
        missing.append(f"JOERN_BIN not found: {run_args.joern_bin}")

    if missing:
        raise FileNotFoundError(" | ".join(missing))


def validate_selector_args(args):
    """
    Validate selector arguments before launching project runs.
    """
    if args.policy == "rl":
        if not args.base_model or not args.lora_path:
            raise ValueError("policy=rl requires --base_model and --lora_path")

    if args.policy == "base":
        if not args.base_model:
            raise ValueError("policy=base requires --base_model")

    if args.policy == "codebert":
        if not args.codebert_model_path:
            raise ValueError("policy=codebert requires --codebert_model_path")
        if not os.path.exists(args.codebert_model_path):
            raise FileNotFoundError(f"CODEBERT_MODEL_PATH not found: {args.codebert_model_path}")

    if args.policy == "qwen_lora":
        if not args.base_model or not args.lora_path:
            raise ValueError("policy=qwen_lora requires --base_model and --lora_path")
        if not os.path.exists(args.base_model):
            raise FileNotFoundError(f"BASE_MODEL not found: {args.base_model}")
        if not os.path.exists(args.lora_path):
            raise FileNotFoundError(f"LORA_PATH not found: {args.lora_path}")
        if args.qwen_lora_max_length <= 0:
            raise ValueError("--qwen_lora_max_length must be greater than 0")
        if not (0.0 <= args.qwen_lora_threshold <= 1.0):
            raise ValueError("--qwen_lora_threshold must be in the [0, 1] range")


def selector_config_summary(args) -> Dict[str, Any]:
    """
    Save this into summary.json / batch_run_summary.json for later experiment alignment.
    """
    config = {
        "policy": args.policy,
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "codebert_model_path": args.codebert_model_path,
        "codebert_threshold": args.codebert_threshold,
        "gpt_model_name": args.gpt_model_name,
        "gpt_base_url": args.gpt_base_url,
    }

    if args.policy == "qwen_lora":
        config.update({
            "qwen_lora_threshold": args.qwen_lora_threshold,
            "qwen_lora_max_length": args.qwen_lora_max_length,
            "qwen_lora_load_in_4bit": not args.qwen_lora_no_4bit,
        })

    return config


# =========================================================
# Core pipeline
# =========================================================

def run_single_chain(
    chain_record: Dict[str, Any],
    inferencer: LayerWiseExploitGoalInferencer,
    path_context_indexer: PathContextIndexer,
    resolver: ExploitStateResolver,
    slicer: Any,
    context_fallback: ContextFallbackBuilder,
    selector,
    sink_requirement_cache: Dict[str, Dict[str, Any]],
    exploit_sketch_dir: str,
    exploit_goal_dir: str,
    execution_plan_dir: str,
    pathcontext_dir: str,
    payload_dir: str,
    save_trace: bool,
    max_steps: int,
    cve_name: str,
) -> Dict[str, Any]:
    chain_id = chain_record["chain_id"]
    chain = chain_record["chain"]
    meta = chain_record.get("meta", {})

    if not chain or not isinstance(chain, list):
        raise ValueError(f"Invalid call chain: {chain_record}")

    sink_method = chain[-1]
    if sink_method not in sink_requirement_cache:
        sink_requirement_cache[sink_method] = inferencer.generate_sink_requirement_from_exploit_sketch(
            exploit_sketch_dir,
            sink_method,
        )
    sink_requirement = sink_requirement_cache[sink_method]

    print("\n" + "=" * 80)
    print(f"[*] Processing {chain_id}")
    print(f"[*] Entry: {chain[0]}")
    print(f"[*] Sink : {sink_method}")
    print("=" * 80)

    start_ts = time.time()
    stem = build_file_stem(chain_id, chain)
    path_context_path = os.path.join(pathcontext_dir, f"{stem}_path_context.json")
    path_context_pair = path_context_indexer.build_path_context_pair(chain, path_context_indexer.cpg_path)
    write_json(path_context_path, {
        "cve_name": cve_name,
        "chain_id": chain_id,
        "path_context_pair": path_context_pair,
    })
    print(f"[*] Path-context pair saved to: {path_context_path}")

    exploit_goal_map = inferencer.infer_layerwise_exploit_goals(path_context_pair, sink_requirement)
    execution_plan = ExecutionPlanBuilder.build_execution_plan(
        chain,
        exploit_goal_map,
        path_context_pair=path_context_pair,
    )
    solve_result = resolver.resolve_with_context_selector(
        execution_plan=execution_plan,
        selector=selector,
        slicer=slicer,
        context_fallback=context_fallback,
        max_steps=max_steps,
        return_trace=save_trace,
        path_context_pair=path_context_pair,
    )

    elapsed_sec = round(time.time() - start_ts, 4)
    final_payload = solve_result["final_payload"]

    exploit_goal_path = os.path.join(exploit_goal_dir, f"{stem}_exploit_goals.json")
    execution_plan_path = os.path.join(execution_plan_dir, f"{stem}_execution_plan.json")
    payload_path = os.path.join(payload_dir, f"{stem}_entry_payload.json")

    write_json(exploit_goal_path, {
        "cve_name": cve_name,
        "chain_id": chain_id,
        "call_chain": chain,
        "sink_method": sink_method,
        "sink_requirement": sink_requirement,
        "path_context_file": path_context_path,
        "exploit_goal_map": exploit_goal_map,
    })

    write_json(execution_plan_path, {
        "cve_name": cve_name,
        "chain_id": chain_id,
        "call_chain": chain,
        "path_context_file": path_context_path,
        "execution_plan": execution_plan,
    })

    payload_obj = {
        "cve_name": cve_name,
        "chain_id": chain_id,
        "call_chain": chain,
        "entry_method": chain[0],
        "sink_method": sink_method,
        "final_payload": final_payload,
        "num_decisions": solve_result["num_decisions"],
        "num_include": solve_result["num_include"],
        "num_exclude": solve_result["num_exclude"],
        "truncated": solve_result["truncated"],
        "elapsed_sec": elapsed_sec,
    }
    if save_trace:
        payload_obj["trace"] = solve_result["trace"]

    write_json(payload_path, payload_obj)

    result_summary = {
        "chain_id": chain_id,
        "entry_method": chain[0],
        "sink_method": sink_method,
        "chain_length": len(chain),
        "elapsed_sec": elapsed_sec,
        "num_decisions": solve_result["num_decisions"],
        "num_include": solve_result["num_include"],
        "num_exclude": solve_result["num_exclude"],
        "truncated": solve_result["truncated"],
        "exploit_goal_file": exploit_goal_path,
        "execution_plan_file": execution_plan_path,
        "goal_map_file": exploit_goal_path,
        "template_file": execution_plan_path,
        "path_context_file": path_context_path,
        "payload_file": payload_path,
        "final_payload": final_payload,
    }
    if meta:
        result_summary["meta"] = meta

    return result_summary


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch run module-2/3/4 pipeline over multiple projects/CVEs and multiple solver models"
    )

    parser.add_argument("--project_cve_map_json", type=str, default=PROJECT_CVE_MAP_JSON)

    parser.add_argument("--cpg_root", type=str, default=CPG_ROOT)
    parser.add_argument("--callchains_root", type=str, default=CALLCHAINS_ROOT)
    parser.add_argument(
        "--exploit_sketch_root",
        type=str,
        default=EXPLOIT_SKETCH_ROOT,
        help="Root directory of sink-level exploit sketches used to generate sink requirements.",
    )
    parser.add_argument("--joern_bin", type=str, default=JOERN_BIN)
    parser.add_argument(
        "--source_root",
        type=str,
        default=None,
        help="Optional client project source root used by PathContextIndexer fallback.",
    )

    parser.add_argument("--solver_models", nargs="*", default=SOLVER_MODELS)

    parser.add_argument(
        "--policy",
        type=str,
        default=POLICY,
        choices=[
            "rl",
            "base",
            "codebert",
            "qwen_lora",
            "gpt",
            "always_include",
            "always_exclude",
            "random",
        ],
    )

    # rl / base / qwen_lora share base_model.
    # rl / qwen_lora share lora_path, but with different meanings:
    #   rl         : CausalLM LoRA
    #   qwen_lora  : SequenceClassification LoRA
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--lora_path", type=str, default=LORA_PATH)

    # CodeBERT selector parameters.
    parser.add_argument("--codebert_model_path", type=str, default=CODEBERT_MODEL_PATH)
    parser.add_argument("--codebert_threshold", type=float, default=CODEBERT_THRESHOLD)

    # Qwen3-8B-LoRA binary selector parameters.
    parser.add_argument("--qwen_lora_threshold", type=float, default=QWEN_LORA_THRESHOLD)
    parser.add_argument("--qwen_lora_max_length", type=int, default=QWEN_LORA_MAX_LENGTH)
    parser.add_argument(
        "--qwen_lora_no_4bit",
        action="store_true",
        default=not QWEN_LORA_LOAD_IN_4BIT,
        help="Disable 4-bit loading for the Qwen3-LoRA selector; useful when VRAM is sufficient or for debugging.",
    )

    # GPT selector parameters.
    parser.add_argument("--gpt_model_name", type=str, default=GPT_MODEL_NAME)
    parser.add_argument("--gpt_base_url", type=str, default=GPT_BASE_URL)

    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save_trace", action="store_true", default=SAVE_TRACE)

    return parser.parse_args()


def run_for_one_project_and_one_solver(
    base_args,
    project_name: str,
    cve_name: str,
    cpg_project: str,
    solver_model: str,
) -> Dict[str, Any]:
    run_args = build_project_run_args(
        base_args=base_args,
        project_name=project_name,
        cve_name=cve_name,
        cpg_project=cpg_project,
        solver_model=solver_model,
    )

    validate_required_inputs(run_args)

    output_base_dir = build_policy_output_dir(run_args)
    execution_plan_dir = os.path.join(output_base_dir, "execution_plans")
    payload_dir = os.path.join(output_base_dir, "entry_payloads")
    exploit_goal_dir = os.path.join(output_base_dir, "exploit_goals")
    pathcontext_dir = os.path.join(output_base_dir, "path_contexts")
    summary_file = os.path.join(output_base_dir, "summary.json")
    debug_prompt_dir = os.path.join(output_base_dir, "debug_prompts")
    temp_dir = os.path.join(output_base_dir, "tmp")

    ensure_dir(output_base_dir)
    ensure_dir(execution_plan_dir)
    ensure_dir(payload_dir)
    ensure_dir(exploit_goal_dir)
    ensure_dir(pathcontext_dir)
    ensure_dir(debug_prompt_dir)
    ensure_dir(temp_dir)

    print("\n" + "#" * 100)
    print(f"[RUN] project_name    = {project_name}")
    print(f"[RUN] cve_name        = {cve_name}")
    print(f"[RUN] cpg_project     = {cpg_project}")
    print(f"[RUN] solver_model    = {solver_model}")
    print(f"[RUN] policy          = {run_args.policy}")
    print(f"[RUN] CPG_FILE        = {run_args.cpg_file}")
    print(f"[RUN] SOURCE_ROOT     = {run_args.source_root}")
    print(f"[RUN] CALLCHAINS_JSON = {run_args.callchains_json}")
    print(f"[RUN] EXPLOIT_SKETCH = {run_args.exploit_sketch_dir}")
    print(f"[RUN] OUTPUT_DIR      = {output_base_dir}")
    print("#" * 100)

    selector = build_selector(
        policy=run_args.policy,
        base_model=run_args.base_model,
        lora_path=run_args.lora_path,
        codebert_model_path=run_args.codebert_model_path,
        codebert_threshold=run_args.codebert_threshold,
        gpt_model_name=run_args.gpt_model_name,
        gpt_base_url=run_args.gpt_base_url,

        # Qwen3-8B-LoRA binary selector parameters.
        qwen_lora_threshold=run_args.qwen_lora_threshold,
        qwen_lora_max_length=run_args.qwen_lora_max_length,
        qwen_lora_load_in_4bit=not run_args.qwen_lora_no_4bit,
    )

    inferencer = LayerWiseExploitGoalInferencer(
        cpg_path=run_args.cpg_file,
        joern_bin=run_args.joern_bin,
        model=run_args.solver_model,
        debug_prompt_dir=debug_prompt_dir,
        base_url=run_args.gpt_base_url,
    )

    resolver = ExploitStateResolver(inferencer)

    path_context_indexer = PathContextIndexer(
        cpg_path=run_args.cpg_file,
        joern_bin=run_args.joern_bin,
        source_root=run_args.source_root,
        inferencer=inferencer,
    )

    slicer = JoernSlicer(
        cpg_path=run_args.cpg_file,
        joern_bin=run_args.joern_bin,
        temp_dir=temp_dir,
    )

    context_fallback = ContextFallbackBuilder()

    chain_records = load_callchains(run_args.callchains_json)
    print(f"[+] Loaded {len(chain_records)} call chains from: {run_args.callchains_json}")

    sink_requirement_cache: Dict[str, Dict[str, Any]] = {}
    all_results: List[Dict[str, Any]] = []

    for chain_record in chain_records:
        result_summary = run_single_chain(
            chain_record=chain_record,
            inferencer=inferencer,
            path_context_indexer=path_context_indexer,
            resolver=resolver,
            slicer=slicer,
            context_fallback=context_fallback,
            selector=selector,
            sink_requirement_cache=sink_requirement_cache,
            exploit_sketch_dir=run_args.exploit_sketch_dir,
            exploit_goal_dir=exploit_goal_dir,
            execution_plan_dir=execution_plan_dir,
            pathcontext_dir=pathcontext_dir,
            payload_dir=payload_dir,
            save_trace=run_args.save_trace,
            max_steps=run_args.max_steps,
            cve_name=cve_name,
        )
        all_results.append(result_summary)

        print(f"[+] chain_id={result_summary['chain_id']}")
        print(json.dumps(result_summary["final_payload"], indent=2, ensure_ascii=False))

    summary = {
        "project_name": project_name,
        "cve_name": cve_name,
        "cpg_project": cpg_project,
        "policy": run_args.policy,
        "selector_config": selector_config_summary(run_args),
        "solver_model": run_args.solver_model,
        "cpg_file": run_args.cpg_file,
        "joern_bin": run_args.joern_bin,
        "callchains_json": run_args.callchains_json,
        "exploit_sketch_dir": run_args.exploit_sketch_dir,
        "output_base_dir": output_base_dir,
        "execution_plan_dir": execution_plan_dir,
        "payload_dir": payload_dir,
        "exploit_goal_dir": exploit_goal_dir,
        "pathcontext_dir": pathcontext_dir,
        "num_chains": len(all_results),
        "results": all_results,
    }

    write_json(summary_file, summary)

    print("\n" + "=" * 80)
    print(f"[+] Done for project={project_name}, solver_model={solver_model}, policy={run_args.policy}")
    print(f"[+] Summary   : {summary_file}")
    print(f"[+] Exploit goals  : {exploit_goal_dir}")
    print(f"[+] Execution plans: {execution_plan_dir}")
    print(f"[+] Path ctx  : {pathcontext_dir}")
    print(f"[+] Payloads  : {payload_dir}")
    print("=" * 80)

    return {
        "project_name": project_name,
        "cve_name": cve_name,
        "cpg_project": cpg_project,
        "solver_model": solver_model,
        "policy": run_args.policy,
        "status": "success",
        "summary_file": summary_file,
        "output_dir": output_base_dir,
        "num_chains": len(all_results),
    }


def main():
    args = parse_args()
    validate_selector_args(args)

    project_items = load_project_cve_map(args.project_cve_map_json)
    solver_models = resolve_solver_models(args)

    print(f"[*] Loaded {len(project_items)} project/CVE items from: {args.project_cve_map_json}")
    print(f"[*] Will run {len(solver_models)} solver model(s): {solver_models}")
    print(f"[*] Selector policy: {args.policy}")
    print(f"[*] Selector config: {json.dumps(selector_config_summary(args), indent=2, ensure_ascii=False)}")

    run_reports: List[Dict[str, Any]] = []

    for item in project_items:
        project_name = item["project"]
        cve_name = item["cve"]
        cpg_project = item["cpg_project"]

        for solver_model in solver_models:
            try:
                report = run_for_one_project_and_one_solver(
                    base_args=args,
                    project_name=project_name,
                    cve_name=cve_name,
                    cpg_project=cpg_project,
                    solver_model=solver_model,
                )
            except Exception as e:
                report = {
                    "project_name": project_name,
                    "cve_name": cve_name,
                    "cpg_project": cpg_project,
                    "solver_model": solver_model,
                    "policy": args.policy,
                    "status": "failed",
                    "error": repr(e),
                }
                print(
                    f"[!] Failed: "
                    f"project={project_name}, "
                    f"cve={cve_name}, "
                    f"cpg_project={cpg_project}, "
                    f"solver_model={solver_model}, "
                    f"policy={args.policy}, "
                    f"error={e}"
                )

            run_reports.append(report)

    batch_summary_path = os.path.join(SCRIPT_DIR, "batch_run_summary.json")
    write_json(batch_summary_path, {
        "project_cve_map_json": args.project_cve_map_json,
        "solver_models": solver_models,
        "policy": args.policy,
        "selector_config": selector_config_summary(args),
        "total_runs": len(run_reports),
        "success_runs": sum(1 for x in run_reports if x["status"] == "success"),
        "failed_runs": sum(1 for x in run_reports if x["status"] == "failed"),
        "reports": run_reports,
    })

    print("\n" + "#" * 100)
    print("[FINAL SUMMARY]")
    print(f"[+] Batch summary saved to: {batch_summary_path}")
    for report in run_reports:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    print("#" * 100)


if __name__ == "__main__":
    main()
