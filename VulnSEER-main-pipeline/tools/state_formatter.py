# tools/state_formatter.py
import json
from typing import Dict, Any, Optional

class StateFormatter:
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def build_prompt(
        layer_goal: Dict[str, Any],
        candidate_item: Dict[str, Any],
        method_code: str,
        already_selected: Dict[str, Any],
        layer_index: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> str:
        goal_str = json.dumps(layer_goal, indent=2, ensure_ascii=False)

        selected_helpers = already_selected.get("helpers", []) or []
        selected_class_context = already_selected.get("class_context", []) or []
        selected_count = len(selected_helpers) + len(selected_class_context)

        selected_token_est = sum(StateFormatter._estimate_tokens(x) for x in selected_helpers + selected_class_context)
        candidate_content = candidate_item.get("content", "") or ""
        candidate_token_est = StateFormatter._estimate_tokens(candidate_content)

        source_type = candidate_item.get("source_type", "unknown")
        priority_hint = "high" if source_type in {"field", "constant", "constructor"} else "medium"
        if source_type in {"helper"}:
            priority_hint = "medium"

        budget_pressure = "low"
        if selected_token_est >= 600:
            budget_pressure = "high"
        elif selected_token_est >= 250:
            budget_pressure = "medium"

        selected_summary = {
            "selected_count": selected_count,
            "selected_token_est": selected_token_est,
            "selected_helper_count": len(selected_helpers),
            "selected_class_context_count": len(selected_class_context),
            "budget_pressure": budget_pressure,
        }

        return f"""You are a context-selection agent in a vulnerability reasoning pipeline.

Task:
Decide whether the current candidate context item should be INCLUDED in the solver prompt.

Action Space:
- Output 1: include this item
- Output 0: exclude this item

Decision Objective:
Select as little context as possible, but keep the most useful evidence for recovering hidden variable origins, constructor effects, constant values, and vulnerability-relevant data flow.

[Current Layer Index]
{layer_index if layer_index is not None else "unknown"}

[Current Goal]
{goal_str}

[Candidate Metadata]
Source Type: {source_type}
Name: {candidate_item.get("name", "unknown")}
Signature: {candidate_item.get("signature", "unknown")}
Priority Hint: {priority_hint}
Estimated New Tokens If Included: {candidate_token_est}

[Candidate Content]
{candidate_content}

[Current Method Code]
```java
{method_code}

[Selected Context Budget Status]
{json.dumps(selected_summary, indent=2, ensure_ascii=False)}

Decision Heuristics:
1.Prefer INCLUDE if this item is likely to reveal:
(1)field initialization
(2)constructor side effects
(3)constant propagation
(4)argument transformation
(5)branch conditions affecting vulnerability trigger
2.Prefer EXCLUDE if this item is:
(1)boilerplate
(2)repeated utility code
(3)generic logging / printing / glue code
(4)unlikely to determine hidden values

Output exactly one character:
0 or 1
"""