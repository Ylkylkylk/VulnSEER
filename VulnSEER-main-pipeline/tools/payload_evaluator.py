# tools/payload_evaluator.py
import json
import os

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

class ExploitPayloadEvaluator:
    """
    LLM-as-a-Judge / lightweight rule-based evaluation.
    """
    # def __init__(self, model="deepseek-chat", use_llm_judge=False):
    #     self.use_llm_judge = use_llm_judge
    #     self.model = model

    #     if self.use_llm_judge:
    #         api_key = os.environ.get("DEEPSEEK_API_KEY")
    #         if not api_key:
    #             raise ValueError("DEEPSEEK_API_KEY is required for judge-model evaluation.")
    #         self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    #     else:
    #         self.client = None
    def __init__(self, model="gpt-5.2", use_llm_judge=False):
        self.use_llm_judge = use_llm_judge
        self.model = model

        if self.use_llm_judge:
            if OpenAI is None:
                raise ImportError("Missing openai Python package; install it first with: pip install openai")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for judge-model evaluation.")
            self.client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")
        else:
            self.client = None

    def evaluate(self, generated_payload: dict, ground_truth_payload: dict) -> float:
        def _safe_dict(v):
            return v if isinstance(v, dict) else {}

        gt_arg_values = _safe_dict(ground_truth_payload.get("arg_values", {}) or {})
        gen_arg_values = _safe_dict(generated_payload.get("arg_values", {}) or {})

        gt_field_values = _safe_dict(ground_truth_payload.get("field_values", {}) or {})
        gen_field_values = _safe_dict(generated_payload.get("field_values", {}) or {})

        gt_arg_count = len(gt_arg_values)
        gen_arg_count = len(gen_arg_values)
        gt_field_count = len(gt_field_values)
        gen_field_count = len(gen_field_values)

        # =========================
        # Stage 1: hard structural gate.
        # =========================
        if gen_arg_count < gt_arg_count or gen_field_count < gt_field_count:
            print(
                "[DEBUG:ExploitPayloadEvaluator] structure_count_mismatch -> "
                f"gt_arg_count={gt_arg_count}, gen_arg_count={gen_arg_count}, "
                f"gt_field_count={gt_field_count}, gen_field_count={gen_field_count}, "
                "final_score=0.00"
            )
            return 0.0

        # =========================
        # Stage 2: score only after structural counts match.
        # =========================
        if not self.use_llm_judge:
            def _norm(v):
                if v is None:
                    return ""
                return str(v).strip().lower()

            def _value_match_ratio(gt_dict, gen_dict):
                if not gt_dict:
                    return 1.0

                hit = 0
                for k, gt_v in gt_dict.items():
                    if k not in gen_dict:
                        continue
                    gv = _norm(gen_dict[k])
                    tv = _norm(gt_v)

                    if not tv:
                        hit += 1 if not gv else 0
                    else:
                        if gv == tv or tv in gv or gv in tv:
                            hit += 1

                return hit / max(1, len(gt_dict))

            # After counts match, check whether keys are covered.
            arg_key_hit = len(set(gt_arg_values.keys()) & set(gen_arg_values.keys())) / max(1, len(gt_arg_values))
            field_key_hit = len(set(gt_field_values.keys()) & set(gen_field_values.keys())) / max(1, len(gt_field_values))

            arg_value_match = _value_match_ratio(gt_arg_values, gen_arg_values)
            field_value_match = _value_match_ratio(gt_field_values, gen_field_values)

            # Evaluate description only after the structural gate passes.
            has_gt_desc = bool((ground_truth_payload.get("description") or "").strip())
            has_gen_desc = bool((generated_payload.get("description") or "").strip())
            description_score = 1.0 if (has_gt_desc and has_gen_desc) else 0.0

            score = (
                0.50 * description_score +
                0.20 * arg_key_hit +
                0.15 * field_key_hit +
                0.10 * arg_value_match +
                0.05 * field_value_match
            )

            print(
                "[DEBUG:ExploitPayloadEvaluator] "
                f"description_score={description_score:.2f}, "
                f"arg_key_hit={arg_key_hit:.2f}, "
                f"field_key_hit={field_key_hit:.2f}, "
                f"arg_value_match={arg_value_match:.2f}, "
                f"field_value_match={field_value_match:.2f}, "
                f"final_score={score:.2f}"
            )
            return max(0.0, min(1.0, score))

        gen_str = json.dumps(generated_payload, indent=2, ensure_ascii=False)
        gt_str = json.dumps(ground_truth_payload, indent=2, ensure_ascii=False)

        prompt = f"""You are an expert security researcher acting as a strict but fair evaluation judge.

    Your task is to compare a Generated Vulnerability Payload against a Ground Truth Payload and assign a score between 0.0 and 1.0.

    Important hard rule:
    1. First compare the number of inputs in arg_values.
    2. Then compare the number of hidden variables in field_values.
    3. If the count of arg_values is NOT exactly the same as the ground truth, the payload must FAIL immediately with score = 0.0.
    4. If the count of field_values is NOT exactly the same as the ground truth, the payload must FAIL immediately with score = 0.0.
    5. Only if BOTH counts are exactly the same, continue to evaluate semantic similarity of description and value compatibility.

    Evaluation order:
    === Step 1: Exact Count Check (Highest Priority) ===
    - Check whether len(generated_payload["arg_values"]) == len(ground_truth_payload["arg_values"])
    - Check whether len(generated_payload["field_values"]) == len(ground_truth_payload["field_values"])
    - If either count mismatches, score must be 0.0

    === Step 2: Description Semantics ===
    Compare the description fields only after the count check passes.
    Focus on whether they describe the same vulnerability-triggering intent, attack goal, exploit condition, and constraints.

    === Step 3: Value Compatibility ===
    Only after the count check passes, compare values in arg_values and field_values.

    === Ground Truth Payload ===
    {gt_str}

    === Generated Payload ===
    {gen_str}

    Output ONLY a JSON object in the following format:
    {{
    "reasoning": "Brief explanation.",
    "arg_count_match": <true or false>,
    "field_count_match": <true or false>,
    "description_similarity": <float between 0.0 and 1.0>,
    "value_compatibility_score": <float between 0.0 and 1.0>,
    "score": <float between 0.0 and 1.0>
    }}
    """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a JSON-only evaluation engine."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)

            arg_count_match = bool(result.get("arg_count_match", False))
            field_count_match = bool(result.get("field_count_match", False))

            if not arg_count_match or not field_count_match:
                print(
                    "[DEBUG:ExploitPayloadEvaluator] "
                    f"arg_count_match={arg_count_match}, "
                    f"field_count_match={field_count_match}, "
                    "final_score=0.00"
                )
                return 0.0

            score = float(result.get("score", 0.0))
            description_similarity = float(result.get("description_similarity", 0.0))
            value_compatibility_score = float(result.get("value_compatibility_score", 0.0))

            print(
                "[DEBUG:ExploitPayloadEvaluator] "
                f"description_similarity={description_similarity:.2f}, "
                f"value_compatibility_score={value_compatibility_score:.2f}, "
                f"final_score={score:.2f}"
            )

            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"[-] Exploit payload evaluator call failed: {e}")
            return 0.0
