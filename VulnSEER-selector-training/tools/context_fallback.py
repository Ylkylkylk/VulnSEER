from typing import Dict, Any

class ContextFallbackBuilder:
    """
    对候选上下文项生成合理的 fallback 文本
    当 item 被纳入 prompt，但 slice 无法获得有效代码时使用
    """

    def generate_fallback_for_context_item(self, item: Dict[str, Any]) -> str:
        source_type = item.get("source_type", "unknown")
        name = item.get("name", "unknown")
        signature = item.get("signature", "unknown")

        if source_type in {"helper", "helper_fallback"}:
            return (
                f"[Fallback Helper Context]\n"
                f"Method `{name}` ({signature}) is assumed to preserve payload propagation "
                f"and not block the vulnerable path."
            )

        if source_type == "external_api":
            return (
                f"[Fallback External API Context]\n"
                f"External API `{name}` ({signature}) is assumed to behave consistently with the vulnerable trigger path."
            )

        if source_type == "field":
            return (
                f"[Fallback Field Context]\n"
                f"Field `{name}` ({signature}) is assumed to be initialized to a valid value "
                f"required by the vulnerable execution path."
            )

        if source_type == "constant":
            return (
                f"[Fallback Constant Context]\n"
                f"Constant `{name}` ({signature}) is assumed to hold a value compatible with payload propagation."
            )

        if source_type == "constructor":
            return (
                f"[Fallback Constructor Context]\n"
                f"Constructor `{signature}` is assumed to initialize internal fields into vulnerability-compatible states."
            )

        if source_type == "class_context_fallback":
            return (
                f"[Fallback Context]\n"
                f"Fallback context for `{signature}` is assumed to contain only vulnerability-compatible setup logic."
            )

        return (
            f"[Fallback Generic Context]\n"
            f"Context item `{name}` ({signature}) is assumed to be compatible with the vulnerable path."
        )
