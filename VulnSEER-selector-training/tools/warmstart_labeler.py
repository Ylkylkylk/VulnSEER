# tools/warmstart_labeler.py
def weak_label_candidate(item: dict) -> int:
    source_type = (item.get("source_type") or "").lower()
    content = (item.get("content") or "").lower()
    name = (item.get("name") or "").lower()
    signature = (item.get("signature") or "").lower()

    low_value_markers = [
        "<empty>", "logger", "println", "printstacktrace",
        "tostring", "hashcode", "equals(", "fallback", "generic context"
    ]
    if any(x in content for x in low_value_markers) or any(x in signature for x in low_value_markers):
        return 0

    if source_type in {"field", "constant", "constructor"}:
        return 1

    useful_helper_markers = [
        "parse", "decode", "encode", "config", "header",
        "path", "url", "token", "password", "request", "response"
    ]
    if source_type == "helper" and (
        any(x in content for x in useful_helper_markers) or
        any(x in name for x in useful_helper_markers)
    ):
        return 1

    return 0