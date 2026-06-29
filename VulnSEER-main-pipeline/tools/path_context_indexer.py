import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple


class PathContextIndexer:
    """
    Materialize the paper-level path-context pair <rho, I_rho> for a known call path.

    The CPG remains the primary source when it is available through
    LayerWiseExploitGoalInferencer.
    Source extraction is used as a deterministic fallback so a path-context pair can
    still be inspected when the local CPG artifact is missing.
    """

    def __init__(
        self,
        cpg_path: Optional[str] = None,
        joern_bin: Optional[str] = None,
        source_root: Optional[str] = None,
        inferencer: Any = None,
    ):
        self.cpg_path = cpg_path
        self.joern_bin = joern_bin
        self.source_root = source_root
        self.inferencer = inferencer
        self._source_cache: Dict[str, Optional[str]] = {}
        self._routine_cache: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}

    def build_path_context_pair(self, chain: List[str], cpg_path: Optional[str] = None) -> Dict[str, Any]:
        if cpg_path:
            self.cpg_path = cpg_path

        routine_contexts: Dict[str, Dict[str, Any]] = {}
        for index, method_fullname in enumerate(chain):
            context_id = f"r{index}"
            routine_contexts[context_id] = self.build_routine_context(
                method_fullname=method_fullname,
                index=index,
                context_id=context_id,
            )

        transition_contexts: Dict[str, Dict[str, Any]] = {}
        for index in range(len(chain) - 1):
            context_id = f"c{index}"
            transition_contexts[context_id] = self.build_transition_context(
                caller=chain[index],
                callee=chain[index + 1],
                caller_context=routine_contexts[f"r{index}"],
                index=index,
                context_id=context_id,
            )

        return {
            "rho": chain,
            "I_rho": {
                "routine_contexts": routine_contexts,
                "transition_contexts": transition_contexts,
            },
            "metadata": {
                "cpg_path": self.cpg_path,
                "source_root": self.source_root,
            },
        }

    def build_routine_context(self, method_fullname: str, index: int, context_id: str) -> Dict[str, Any]:
        cache_key = (self.cpg_path, method_fullname)
        if cache_key in self._routine_cache:
            return self._deepcopy_jsonish(self._routine_cache[cache_key])

        cpg_info = self._get_cpg_method_context(method_fullname)
        source_info = self._get_source_method_context(method_fullname)

        if cpg_info.get("method_found"):
            method_info = self._merge_method_info(cpg_info, source_info)
            extraction_backend = "cpg"
        elif source_info.get("method_found"):
            method_info = self._merge_method_info(source_info, cpg_info)
            extraction_backend = "source_fallback"
        else:
            method_info = self._external_method_context(method_fullname)
            method_info = self._merge_method_info(method_info, source_info)
            extraction_backend = "external_or_missing"

        parsed = self._parse_method_fullname(method_fullname)
        params = method_info.get("params", []) or []
        code = method_info.get("code", "") or ""

        routine_context = {
            "method": method_fullname,
            "declaring_type": parsed["class_fullname"],
            "name": parsed["method_name"],
            "parameters": params,
            "source_location": method_info.get("source_location", {}),
            "receiver_type": method_info.get("receiver_type") or parsed["class_fullname"],
            "source_code": code,
            "field_accesses": self._compact_field_items(method_info.get("field_accesses", [])),
            "constants": self._compact_field_items(method_info.get("constants", [])),
            "literals": method_info.get("literals", []),
            "constructors": self._compact_constructor_items(method_info.get("constructors", [])),
            "local_helpers": self._compact_helper_items(method_info.get("helpers", [])),
            "class_hierarchy": method_info.get("class_hierarchy", {}),
            "object_definitions": self._compact_object_items(method_info.get("object_definitions", [])),
            "types": {
                "locals": method_info.get("local_variable_types", {}),
                "fields": method_info.get("field_types", {}),
            },
            "method_found": bool(method_info.get("method_found")),
            "is_external": bool(method_info.get("is_external")),
            "extraction_backend": extraction_backend,
        }
        if method_info.get("no_context_reason"):
            routine_context["missing_reason"] = method_info["no_context_reason"]
        routine_context = self._drop_empty(routine_context)

        self._routine_cache[cache_key] = self._deepcopy_jsonish(routine_context)
        return routine_context

    def build_transition_context(
        self,
        caller: str,
        callee: str,
        caller_context: Dict[str, Any],
        index: int,
        context_id: str,
    ) -> Dict[str, Any]:
        source_transition = self._extract_transition_from_source(caller, callee, caller_context)
        cpg_transition = self._get_cpg_transition_context(caller, callee)
        transition = self._merge_transition_info(cpg_transition, source_transition)
        compact = {
            "caller": caller,
            "callee": callee,
            "call_site": transition.get("call_site") or transition.get("successor_call_site", ""),
            "call_site_location": transition.get("call_site_location", {}),
            "receiver_expression": transition.get("receiver_expression", ""),
            "actual_arguments": transition.get("actual_parameter_expressions", []),
            "enclosing_conditions": transition.get("enclosing_branch_conditions", []),
            "nearby_slice": transition.get("nearby_code_slice", ""),
            "propagation_context": transition.get("propagation_context", []),
            "dynamic_dispatch": transition.get("dynamic_dispatch_targets", []),
            "extraction_backend": transition.get("extraction_backend", ""),
            "source_enrichment_backend": transition.get("source_enrichment_backend", ""),
        }
        if transition.get("no_context_reason"):
            compact["missing_reason"] = transition["no_context_reason"]
        return self._drop_empty(compact)

    def _get_cpg_transition_context(self, caller: str, callee: str) -> Dict[str, Any]:
        if not self.cpg_path or not self.joern_bin:
            return {}
        if not os.path.exists(self.cpg_path) or not os.path.exists(self.joern_bin):
            return {}

        parsed = self._parse_method_fullname(callee)
        safe_cpg = self.cpg_path.replace('"', '\\"')
        safe_caller = caller.replace('"', '\\"')
        safe_callee = callee.replace('"', '\\"')
        safe_name = parsed["method_name"].replace('"', '\\"')
        safe_class = parsed["class_simple"].replace('"', '\\"')
        expected_arg_count = parsed.get("arg_count")
        expected_arg_count_str = "None" if expected_arg_count is None else f"Some({expected_arg_count})"

        scala_script = f'''
importCpg("{safe_cpg}")
val callerOpt = cpg.method.fullNameExact("{safe_caller}").headOption
val calleeFull = "{safe_callee}"
val calleeName = "{safe_name}"
val calleeClass = "{safe_class}"
val expectedArgCount: Option[Int] = {expected_arg_count_str}
println("===TRANSITION_START===")
if (callerOpt.isDefined) {{
  val calls = callerOpt.get.call.l.filter {{ c =>
    val code = Option(c.code).getOrElse("")
    val name = Option(c.name).getOrElse("")
    val fullName = Option(c.methodFullName).getOrElse("")
    val args = c.argument.sortBy(_.order).map(_.code).filterNot(_ == null)
    val argCountOk = expectedArgCount.map(n => args.size == n || args.size == n + 1).getOrElse(true)
    val nameHit =
      fullName == calleeFull ||
      name == calleeName ||
      (calleeName == "<init>" && code.contains("new " + calleeClass + "(")) ||
      code.contains("." + calleeName + "(") ||
      code.contains(calleeName + "(")
    nameHit && argCountOk
  }}.take(20)
  if (calls.nonEmpty) {{
    val c = calls.head
    println("CODE:" + Option(c.code).getOrElse(""))
    println("NAME:" + Option(c.name).getOrElse(""))
    println("METHOD_FULL_NAME:" + Option(c.methodFullName).getOrElse(""))
    println("LINE:" + c.lineNumber.map(_.toString).getOrElse(""))
    println("COLUMN:" + c.columnNumber.map(_.toString).getOrElse(""))
    println("ARGS_START")
    c.argument.sortBy(_.order).foreach(a => println(a.order.toString + ":::" + Option(a.code).getOrElse("")))
    println("ARGS_END")
  }} else {{
    println("NO_CALL_FOUND")
  }}
}} else {{
  println("NO_CALLER_FOUND")
}}
println("===TRANSITION_END===")
exit
'''

        env = os.environ.copy()
        if env.get("JAVA_HOME"):
            env["PATH"] = f"{env['JAVA_HOME']}/bin:" + env.get("PATH", "")

        try:
            result = subprocess.run(
                [self.joern_bin],
                input=scala_script,
                capture_output=True,
                text=True,
                timeout=90,
                env=env,
            )
        except Exception as exc:
            return {
                "extraction_backend": "cpg_failed",
                "no_context_reason": f"CPG transition extraction failed: {exc}",
            }

        stdout = result.stdout or ""
        start = stdout.find("===TRANSITION_START===")
        end = stdout.find("===TRANSITION_END===")
        if start == -1 or end == -1:
            return {
                "extraction_backend": "cpg_failed",
                "no_context_reason": "CPG transition extraction did not emit transition markers.",
            }

        body = stdout[start + len("===TRANSITION_START==="):end].strip()
        if "NO_CALL_FOUND" in body or "NO_CALLER_FOUND" in body:
            return {}

        parsed_body = self._parse_keyed_section(body)
        arg_lines = self._extract_section(body, "ARGS_START", "ARGS_END")
        actual_args = self._parse_cpg_args(arg_lines, expected_arg_count)
        call_site = parsed_body.get("CODE", "")
        return {
            "successor_call_site": call_site,
            "call_site": call_site,
            "call_site_location": {
                "line": self._safe_int(parsed_body.get("LINE")),
                "column": self._safe_int(parsed_body.get("COLUMN")),
            },
            "receiver_expression": self._infer_receiver_from_cpg_args(actual_args, call_site),
            "actual_parameter_expressions": actual_args,
            "dynamic_dispatch_targets": [{
                "target_method": callee,
                "cpg_method_full_name": parsed_body.get("METHOD_FULL_NAME", ""),
                "cpg_call_name": parsed_body.get("NAME", ""),
                "dispatch_kind": "constructor" if parsed["method_name"] == "<init>" else "method_call",
            }],
            "extraction_backend": "cpg",
        }

    def _get_cpg_method_context(self, method_fullname: str) -> Dict[str, Any]:
        if not self.inferencer:
            return {}
        if self.cpg_path and not os.path.exists(self.cpg_path):
            return {}
        try:
            return self.inferencer.get_method_context_from_cpg(method_fullname)
        except Exception as exc:
            return {
                "method_found": False,
                "is_external": False,
                "no_context_reason": f"CPG routine extraction failed: {exc}",
            }

    def _get_source_method_context(self, method_fullname: str) -> Dict[str, Any]:
        parsed = self._parse_method_fullname(method_fullname)
        source_path = self._resolve_java_source_path(parsed["top_level_class"])
        if not source_path:
            return {"method_found": False}

        source = self._read_text(source_path)
        if source is None:
            return {"method_found": False}

        method_name = parsed["method_name"]
        lookup_name = parsed["class_simple"] if method_name == "<init>" else method_name
        method_block = self._find_method_block(
            source=source,
            method_name=lookup_name,
            arg_count=parsed["arg_count"],
        )
        if not method_block:
            return {
                "method_found": False,
                "source_location": {"file": source_path},
            }

        code = method_block["code"]
        params = self._parse_source_params(method_block["params"])
        fields = self._extract_fields(source, method_fullname)
        constructors = self._extract_constructors(source, parsed["class_simple"], source_path)
        helpers = self._extract_local_helpers(source, code, method_fullname, parsed["class_fullname"])
        literals = self._extract_literals(code)
        object_defs = self._extract_object_definitions(code)
        local_types = self._extract_local_variable_types(code)
        field_types = {
            item["name"]: item.get("type", "")
            for item in fields
            if item.get("name")
        }

        class_context = []
        for item in fields:
            class_context.append({
                "item_id": item["item_id"],
                "source_type": item["source_type"],
                "name": item["name"],
                "signature": item["signature"],
                "content": item["content"],
                "priority_hint": item.get("priority_hint", "high"),
            })
        for ctor in constructors:
            class_context.append({
                "item_id": ctor["item_id"],
                "source_type": "constructor",
                "name": "<init>",
                "signature": ctor["signature"],
                "content": ctor["content"],
                "priority_hint": "high",
            })
        for idx, obj in enumerate(object_defs):
            class_context.append({
                "item_id": f"object_definition_{idx}",
                "source_type": "object_definition",
                "name": obj.get("type", f"object_{idx}"),
                "signature": method_fullname,
                "content": obj.get("code", ""),
                "priority_hint": "medium",
            })

        class_hierarchy = self._extract_class_hierarchy(source, parsed["class_simple"])
        if class_hierarchy.get("declaration"):
            class_context.append({
                "item_id": "class_hierarchy_0",
                "source_type": "class_hierarchy",
                "name": parsed["class_simple"],
                "signature": parsed["class_fullname"],
                "content": class_hierarchy["declaration"],
                "priority_hint": "medium",
            })

        return {
            "code": code,
            "params": params,
            "helpers": helpers,
            "class_context": class_context,
            "method_found": True,
            "is_external": False,
            "no_context_reason": "",
            "source_location": {
                "file": source_path,
                "start_line": method_block["start_line"],
                "end_line": method_block["end_line"],
            },
            "receiver_type": parsed["class_fullname"],
            "contextual_receiver_type": parsed["class_fullname"],
            "field_accesses": self._extract_field_accesses(code, fields),
            "constants": [x for x in fields if x.get("source_type") == "constant"],
            "literals": literals,
            "constructors": constructors,
            "class_hierarchy": class_hierarchy,
            "object_definitions": object_defs,
            "local_variable_types": local_types,
            "field_types": field_types,
        }

    def _extract_transition_from_source(
        self,
        caller: str,
        callee: str,
        caller_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        parsed_callee = self._parse_method_fullname(callee)
        code = caller_context.get("source_code") or caller_context.get("code", "") or ""
        source_location = caller_context.get("source_location", {}) or {}
        call = self._find_call_in_code(code, parsed_callee)

        transition = {
            "successor_call_site": "",
            "call_site": "",
            "call_site_location": {},
            "receiver_expression": "",
            "actual_parameter_expressions": [],
            "enclosing_branch_conditions": [],
            "nearby_code_slice": "",
            "dynamic_dispatch_targets": [],
            "propagation_context": [],
            "extraction_backend": "source_fallback",
        }

        if not call:
            transition["no_context_reason"] = "Call site was not found in materialized caller source."
            transition["dynamic_dispatch_targets"] = [{
                "target_method": callee,
                "reason": "Known path edge from rho; call-site source unavailable.",
            }]
            return transition

        start_line = source_location.get("start_line", 1) or 1
        absolute_line = start_line + call["line_index"]
        receiver_type = self._resolve_receiver_type(call["receiver_expression"], caller_context)

        transition.update({
            "successor_call_site": call["call_site"],
            "call_site": call["call_site"],
            "call_site_location": {
                "file": source_location.get("file", ""),
                "line": absolute_line,
            },
            "receiver_expression": call["receiver_expression"],
            "actual_parameter_expressions": call["actual_parameter_expressions"],
            "enclosing_branch_conditions": self._collect_enclosing_conditions(code, call["line_index"]),
            "nearby_code_slice": self._nearby_code_slice(code, call["line_index"]),
            "dynamic_dispatch_targets": [{
                "target_method": callee,
                "receiver_expression": call["receiver_expression"],
                "declared_receiver_type": receiver_type,
                "dispatch_kind": "constructor" if parsed_callee["method_name"] == "<init>" else "method_call",
            }],
            "propagation_context": self._collect_propagation_context(
                code=code,
                line_index=call["line_index"],
                receiver_expression=call["receiver_expression"],
                actual_args=call["actual_parameter_expressions"],
            ),
            "extraction_backend": "source_fallback",
        })
        return transition

    def _find_call_in_code(self, code: str, parsed_callee: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not code:
            return None

        callee_name = parsed_callee["method_name"]
        search_names = []
        if callee_name == "<init>":
            search_names.append(parsed_callee["class_simple"])
        else:
            search_names.append(callee_name)
        search_names = [x for x in search_names if x]

        for name in search_names:
            pattern = re.compile(r"\b" + re.escape(name) + r"\s*\(")
            for match in pattern.finditer(code):
                if match.start() > 0 and code[match.start() - 1] == ".":
                    receiver = self._extract_receiver_before_call(code, match.start())
                else:
                    receiver = self._extract_receiver_before_call(code, match.start())

                if not self._looks_like_call_site(code, match.start(), name, parsed_callee):
                    continue

                open_paren = code.find("(", match.start())
                close_paren = self._find_matching(code, open_paren, "(", ")")
                if close_paren == -1:
                    continue

                args_text = code[open_paren + 1:close_paren]
                actual_args = self._split_args(args_text)
                if (
                    parsed_callee.get("arg_count") is not None
                    and len(actual_args) != parsed_callee.get("arg_count")
                ):
                    continue

                line_index = code[:match.start()].count("\n")
                line = code.splitlines()[line_index].strip()

                return {
                    "line_index": line_index,
                    "call_site": line,
                    "receiver_expression": receiver,
                    "actual_parameter_expressions": actual_args,
                }
        return None

    def _looks_like_call_site(
        self,
        code: str,
        match_start: int,
        name: str,
        parsed_callee: Dict[str, Any],
    ) -> bool:
        line_start = code.rfind("\n", 0, match_start) + 1
        line_prefix = code[line_start:match_start]

        if parsed_callee["method_name"] == "<init>":
            return "new " in line_prefix or name in parsed_callee["class_simple"]

        declaration_markers = [" public ", " private ", " protected ", " static ", " final "]
        declaration_prefix = " " + line_prefix.strip()
        if any(marker in declaration_prefix for marker in declaration_markers) and "." not in line_prefix:
            return False

        return True

    def _extract_receiver_before_call(self, code: str, match_start: int) -> str:
        prefix = code[max(0, match_start - 120):match_start]
        new_match = re.search(r"new\s+([A-Za-z_][\w.$<>]*)\s*$", prefix)
        if new_match:
            return "new " + new_match.group(1)

        receiver_match = re.search(r"([A-Za-z_][\w.$]*(?:\([^)]*\))?)\s*\.\s*$", prefix)
        if receiver_match:
            return receiver_match.group(1)

        return ""

    def _resolve_receiver_type(self, receiver_expression: str, caller_context: Dict[str, Any]) -> str:
        if not receiver_expression:
            return caller_context.get("declaring_type", "")
        if receiver_expression.startswith("new "):
            return receiver_expression.replace("new ", "", 1).strip()

        name = receiver_expression.split(".")[-1].strip()
        types = caller_context.get("types", {}) or {}
        local_types = caller_context.get("local_variable_types", {}) or types.get("locals", {}) or {}
        field_types = caller_context.get("field_types", {}) or types.get("fields", {}) or {}
        return local_types.get(name) or field_types.get(name) or ""

    def _collect_enclosing_conditions(self, code: str, line_index: int) -> List[str]:
        conditions = []
        lines = code.splitlines()
        for line in lines[:line_index + 1]:
            stripped = line.strip()
            if re.match(r"^(if|switch|for|while|catch)\s*\(", stripped) or stripped.startswith("case "):
                conditions.append(stripped)
        return conditions[-6:]

    def _nearby_code_slice(self, code: str, line_index: int, radius: int = 3) -> str:
        lines = code.splitlines()
        start = max(0, line_index - radius)
        end = min(len(lines), line_index + radius + 1)
        return "\n".join(lines[start:end]).strip()

    def _collect_propagation_context(
        self,
        code: str,
        line_index: int,
        receiver_expression: str,
        actual_args: List[str],
    ) -> List[str]:
        symbols = set()
        for item in [receiver_expression] + actual_args:
            for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", item or ""):
                if token not in {"new", "return", "null", "true", "false"}:
                    symbols.add(token)

        if not symbols:
            return []

        lines = code.splitlines()
        lower = max(0, line_index - 12)
        upper = min(len(lines), line_index + 4)
        context = []
        for line in lines[lower:upper]:
            stripped = line.strip()
            if stripped and any(re.search(r"\b" + re.escape(sym) + r"\b", stripped) for sym in symbols):
                context.append(stripped)
        return context

    def _merge_method_info(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(primary or {})
        for key, value in (secondary or {}).items():
            if key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value

        if not merged.get("helpers") and secondary.get("helpers"):
            merged["helpers"] = secondary["helpers"]
        if not merged.get("class_context") and secondary.get("class_context"):
            merged["class_context"] = secondary["class_context"]
        if not merged.get("params") and secondary.get("params"):
            merged["params"] = secondary["params"]

        return merged

    def _merge_transition_info(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        if not primary:
            return dict(secondary or {})

        merged = dict(primary)
        for key, value in (secondary or {}).items():
            if key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value

        if primary.get("extraction_backend") == "cpg" and secondary:
            merged["source_enrichment_backend"] = secondary.get("extraction_backend")
            if secondary.get("actual_parameter_expressions"):
                if primary.get("actual_parameter_expressions") != secondary.get("actual_parameter_expressions"):
                    merged["cpg_argument_expressions"] = primary.get("actual_parameter_expressions", [])
                merged["actual_parameter_expressions"] = secondary.get("actual_parameter_expressions", [])
            if secondary.get("receiver_expression") not in (None, ""):
                if primary.get("receiver_expression") != secondary.get("receiver_expression"):
                    merged["cpg_receiver_expression"] = primary.get("receiver_expression", "")
                merged["receiver_expression"] = secondary.get("receiver_expression", "")
            if secondary.get("call_site") and len(secondary.get("call_site", "")) > len(primary.get("call_site", "")):
                merged["cpg_call_site"] = primary.get("call_site", "")
                merged["call_site"] = secondary.get("call_site", "")
                merged["successor_call_site"] = secondary.get("successor_call_site", secondary.get("call_site", ""))
        return merged

    def _parse_keyed_section(self, text: str) -> Dict[str, str]:
        values = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key.isupper():
                values[key] = value.strip()
        return values

    def _extract_section(self, text: str, start_tag: str, end_tag: str) -> str:
        start = text.find(start_tag)
        end = text.find(end_tag)
        if start == -1 or end == -1 or end < start:
            return ""
        return text[start + len(start_tag):end].strip()

    def _parse_cpg_args(self, arg_lines: str, expected_arg_count: Optional[int]) -> List[str]:
        if expected_arg_count == 0:
            return []

        parsed_args = []
        for line in arg_lines.splitlines():
            if ":::" not in line:
                continue
            order_text, code = line.split(":::", 1)
            try:
                order = int(order_text.strip())
            except ValueError:
                order = len(parsed_args) + 1
            parsed_args.append((order, code.strip()))

        parsed_args.sort(key=lambda item: item[0])
        args = [code for _, code in parsed_args if code]
        if expected_arg_count is not None and len(args) == expected_arg_count + 1:
            args = args[-expected_arg_count:]
        return args

    def _infer_receiver_from_cpg_args(self, actual_args: List[str], call_site: str) -> str:
        if call_site.startswith("new "):
            match = re.match(r"new\s+([A-Za-z_][\w.$<>]*)", call_site)
            return "new " + match.group(1) if match else ""

        receiver_match = re.search(r"([A-Za-z_][\w.$]*(?:\([^)]*\))?)\s*\.\s*[A-Za-z_][\w$]*\s*\(", call_site)
        if receiver_match:
            return receiver_match.group(1)

        if actual_args and "." in call_site:
            return actual_args[0]
        return ""

    def _safe_int(self, value: Optional[str]) -> Optional[int]:
        try:
            return int(value) if value not in (None, "") else None
        except ValueError:
            return None

    def _compact_helper_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in items or []:
            compact.append(self._drop_empty({
                "name": item.get("name", ""),
                "signature": item.get("signature", ""),
                "source_code": item.get("source_code") or item.get("content", ""),
            }))
        return [x for x in compact if x]

    def _compact_field_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in items or []:
            compact.append(self._drop_empty({
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "signature": item.get("signature", ""),
                "code": item.get("code") or item.get("source") or item.get("content", ""),
            }))
        return [x for x in compact if x]

    def _compact_constructor_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in items or []:
            compact.append(self._drop_empty({
                "signature": item.get("signature", ""),
                "source_code": item.get("source_code") or item.get("content", ""),
                "source_location": item.get("source_location", {}),
            }))
        return [x for x in compact if x]

    def _compact_object_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in items or []:
            compact.append(self._drop_empty({
                "type": item.get("type", ""),
                "code": item.get("code", ""),
            }))
        return [x for x in compact if x]

    def _drop_empty(self, value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                cleaned = self._drop_empty(item)
                if cleaned in (None, "", [], {}):
                    continue
                result[key] = cleaned
            return result
        if isinstance(value, list):
            result = []
            for item in value:
                cleaned = self._drop_empty(item)
                if cleaned in (None, "", [], {}):
                    continue
                result.append(cleaned)
            return result
        return value

    def _external_method_context(self, method_fullname: str) -> Dict[str, Any]:
        return {
            "code": "",
            "params": [],
            "helpers": [],
            "class_context": [],
            "method_found": False,
            "is_external": True,
            "no_context_reason": "Method not found in project CPG/source; likely external library method.",
        }

    def _resolve_java_source_path(self, top_level_class: str) -> Optional[str]:
        if not self.source_root:
            return None
        if top_level_class in self._source_cache:
            return self._source_cache[top_level_class]

        rel_path = top_level_class.replace(".", os.sep) + ".java"
        roots = [
            self.source_root,
            os.path.join(self.source_root, "src", "main", "java"),
            os.path.join(self.source_root, "src", "test", "java"),
        ]
        for root in roots:
            candidate = os.path.join(root, rel_path)
            if os.path.exists(candidate):
                self._source_cache[top_level_class] = candidate
                return candidate

        class_name = top_level_class.rsplit(".", 1)[-1] + ".java"
        for root, _, files in os.walk(self.source_root):
            if class_name in files:
                candidate = os.path.join(root, class_name)
                self._source_cache[top_level_class] = candidate
                return candidate

        self._source_cache[top_level_class] = None
        return None

    def _read_text(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return None

    def _find_method_block(self, source: str, method_name: str, arg_count: Optional[int]) -> Optional[Dict[str, Any]]:
        pattern = re.compile(r"\b" + re.escape(method_name) + r"\s*\(")
        for match in pattern.finditer(source):
            if match.start() > 0 and source[match.start() - 1] == ".":
                continue

            open_paren = source.find("(", match.start())
            close_paren = self._find_matching(source, open_paren, "(", ")")
            if close_paren == -1:
                continue

            params_text = source[open_paren + 1:close_paren]
            if arg_count is not None and len(self._split_args(params_text)) != arg_count:
                continue

            brace = source.find("{", close_paren)
            semicolon = source.find(";", close_paren)
            if brace == -1 or (semicolon != -1 and semicolon < brace):
                continue

            line_start = source.rfind("\n", 0, match.start()) + 1
            declaration_prefix = source[line_start:match.start()]
            if " return " in " " + declaration_prefix or "=" in declaration_prefix:
                continue

            block_end = self._find_matching(source, brace, "{", "}")
            if block_end == -1:
                continue

            method_start = self._include_annotations_start(source, line_start)
            code = source[method_start:block_end + 1].strip()
            return {
                "code": code,
                "params": params_text,
                "start_line": source[:method_start].count("\n") + 1,
                "end_line": source[:block_end].count("\n") + 1,
            }
        return None

    def _include_annotations_start(self, source: str, line_start: int) -> int:
        current = line_start
        while current > 0:
            prev_end = current - 1
            prev_start = source.rfind("\n", 0, prev_end) + 1
            prev_line = source[prev_start:prev_end].strip()
            if not prev_line:
                current = prev_start
                continue
            if prev_line.startswith("@"):
                current = prev_start
                continue
            break
        return current

    def _find_matching(self, text: str, start: int, open_char: str, close_char: str) -> int:
        if start < 0 or start >= len(text) or text[start] != open_char:
            return -1

        depth = 0
        in_string = False
        in_char = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not in_char:
                in_string = not in_string
                continue
            if ch == "'" and not in_string:
                in_char = not in_char
                continue
            if in_string or in_char:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _parse_source_params(self, params_text: str) -> List[Dict[str, str]]:
        params = []
        for idx, raw in enumerate(self._split_args(params_text)):
            cleaned = re.sub(r"@\w+(?:\([^)]*\))?", "", raw).strip()
            cleaned = re.sub(r"\b(final|volatile)\b", "", cleaned).strip()
            if not cleaned:
                continue
            parts = cleaned.split()
            if len(parts) == 1:
                params.append({"name": f"arg{idx}", "type": parts[0].replace("...", "[]")})
            else:
                params.append({
                    "name": parts[-1].replace("...", "").strip(),
                    "type": " ".join(parts[:-1]).replace("...", "[]").strip(),
                })
        return params

    def _split_args(self, args_text: str) -> List[str]:
        args = []
        current = []
        depth = 0
        in_string = False
        in_char = False
        escape = False
        for ch in args_text:
            if escape:
                current.append(ch)
                escape = False
                continue
            if ch == "\\":
                current.append(ch)
                escape = True
                continue
            if ch == '"' and not in_char:
                in_string = not in_string
            elif ch == "'" and not in_string:
                in_char = not in_char
            elif not in_string and not in_char:
                if ch in "([{<":
                    depth += 1
                elif ch in ")]}>":
                    depth = max(0, depth - 1)
                elif ch == "," and depth == 0:
                    arg = "".join(current).strip()
                    if arg:
                        args.append(arg)
                    current = []
                    continue
            current.append(ch)
        arg = "".join(current).strip()
        if arg:
            args.append(arg)
        return args

    def _extract_fields(self, source: str, method_fullname: str) -> List[Dict[str, Any]]:
        fields = []
        field_idx = 0
        for match in re.finditer(
            r"^\s*(?P<mods>(?:public|protected|private|static|final|transient|volatile)\s+)*"
            r"(?P<type>[A-Za-z_][\w.$<>\[\], ?]*)\s+"
            r"(?P<name>[A-Za-z_][\w]*)\s*(?P<init>=\s*[^;]+)?;",
            source,
            flags=re.MULTILINE,
        ):
            line = match.group(0).strip()
            if "(" in line or ")" in line:
                continue
            source_type = "constant" if "static" in line and "final" in line else "field"
            fields.append({
                "item_id": f"{source_type}_{field_idx}",
                "source_type": source_type,
                "name": match.group("name"),
                "type": match.group("type").strip(),
                "signature": f"{method_fullname}#{match.group('name')}",
                "content": line,
                "priority_hint": "high" if source_type == "constant" else "medium",
            })
            field_idx += 1
        return fields[:30]

    def _extract_constructors(self, source: str, class_simple: str, source_path: str) -> List[Dict[str, Any]]:
        constructors = []
        for idx, match in enumerate(re.finditer(r"\b" + re.escape(class_simple) + r"\s*\(", source)):
            if match.start() > 0 and source[match.start() - 1] == ".":
                continue
            open_paren = source.find("(", match.start())
            close_paren = self._find_matching(source, open_paren, "(", ")")
            brace = source.find("{", close_paren)
            if close_paren == -1 or brace == -1:
                continue
            block_end = self._find_matching(source, brace, "{", "}")
            if block_end == -1:
                continue
            line_start = source.rfind("\n", 0, match.start()) + 1
            code = source[line_start:block_end + 1].strip()
            if " new " in " " + source[line_start:match.start()]:
                continue
            constructors.append({
                "item_id": f"ctor_{idx}",
                "signature": f"{class_simple}.<init>",
                "content": code,
                "source_location": {
                    "file": source_path,
                    "start_line": source[:line_start].count("\n") + 1,
                    "end_line": source[:block_end].count("\n") + 1,
                },
            })
        return constructors[:8]

    def _extract_local_helpers(
        self,
        source: str,
        method_code: str,
        method_fullname: str,
        class_fullname: str,
    ) -> List[Dict[str, Any]]:
        keywords = {
            "if", "for", "while", "switch", "catch", "return", "throw", "new",
            "super", "this", "try", "synchronized",
        }
        helper_names = []
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", method_code):
            if name in keywords:
                continue
            if name and name not in helper_names:
                helper_names.append(name)

        helpers = []
        for idx, name in enumerate(helper_names[:20]):
            block = self._find_method_block(source, name, arg_count=None)
            if not block:
                continue
            if block["code"] == method_code:
                continue
            helpers.append({
                "item_id": f"helper_{idx}",
                "source_type": "helper",
                "name": name,
                "signature": f"{class_fullname}.{name}",
                "content": block["code"],
                "priority_hint": "medium",
            })
        return helpers

    def _extract_literals(self, code: str) -> List[Dict[str, Any]]:
        literals = []
        seen = set()
        for idx, value in enumerate(re.findall(r'"(?:\\.|[^"\\])*"', code)):
            if value in seen:
                continue
            seen.add(value)
            literals.append({"kind": "string", "value": value, "item_id": f"literal_string_{idx}"})
        for idx, value in enumerate(re.findall(r"(?<![\w.])-?\b\d+(?:\.\d+)?\b", code)):
            if value in seen:
                continue
            seen.add(value)
            literals.append({"kind": "number", "value": value, "item_id": f"literal_number_{idx}"})
        for value in ["true", "false", "null"]:
            if re.search(r"\b" + value + r"\b", code):
                literals.append({"kind": "keyword", "value": value, "item_id": f"literal_{value}"})
        return literals[:40]

    def _extract_object_definitions(self, code: str) -> List[Dict[str, str]]:
        objects = []
        for idx, match in enumerate(re.finditer(r"new\s+([A-Za-z_][\w.$<>]*)\s*\(", code)):
            line_start = code.rfind("\n", 0, match.start()) + 1
            line_end = code.find("\n", match.start())
            if line_end == -1:
                line_end = len(code)
            objects.append({
                "type": match.group(1),
                "code": code[line_start:line_end].strip(),
                "item_id": f"object_definition_{idx}",
            })
        return objects[:20]

    def _extract_local_variable_types(self, code: str) -> Dict[str, str]:
        local_types = {}
        for match in re.finditer(
            r"(?:^|[\n;{])\s*(?:final\s+)?(?P<type>[A-Z][A-Za-z0-9_.$<>]*)\s+"
            r"(?P<name>[a-zA-Z_][A-Za-z0-9_]*)\s*(?:=|;)",
            code,
        ):
            local_types[match.group("name")] = match.group("type")
        return local_types

    def _extract_field_accesses(self, code: str, fields: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        result = []
        for field in fields:
            name = field.get("name", "")
            if name and re.search(r"\b" + re.escape(name) + r"\b", code):
                result.append({
                    "name": name,
                    "type": field.get("type", ""),
                    "source": field.get("content", ""),
                })
        for name in re.findall(r"\bthis\.([A-Za-z_][A-Za-z0-9_]*)\b", code):
            if not any(item.get("name") == name for item in result):
                result.append({"name": name, "type": "", "source": "this." + name})
        return result

    def _extract_class_hierarchy(self, source: str, class_simple: str) -> Dict[str, Any]:
        pattern = re.compile(
            r"(?P<decl>(?:public\s+|protected\s+|private\s+|abstract\s+|final\s+)*"
            r"(?:class|interface|enum)\s+" + re.escape(class_simple) +
            r"(?:\s+extends\s+(?P<extends>[A-Za-z0-9_.$<>]+))?"
            r"(?:\s+implements\s+(?P<implements>[A-Za-z0-9_.$<>,\s]+))?)"
        )
        match = pattern.search(source)
        if not match:
            return {}
        implements = []
        if match.group("implements"):
            implements = [x.strip() for x in match.group("implements").split(",") if x.strip()]
        return {
            "declaration": match.group("decl").strip(),
            "extends": match.group("extends") or "",
            "implements": implements,
        }

    def _parse_method_fullname(self, method_fullname: str) -> Dict[str, Any]:
        before_colon = method_fullname.split(":", 1)[0]
        if "." in before_colon:
            class_fullname, method_name = before_colon.rsplit(".", 1)
        else:
            class_fullname, method_name = "", before_colon

        top_level_class = class_fullname.split("$", 1)[0]
        class_simple = top_level_class.rsplit(".", 1)[-1] if top_level_class else ""
        arg_count = None
        args_match = re.search(r"\((.*)\)\s*$", method_fullname)
        if args_match:
            args_text = args_match.group(1).strip()
            arg_count = 0 if not args_text else len(self._split_args(args_text))

        return {
            "class_fullname": class_fullname,
            "top_level_class": top_level_class,
            "class_simple": class_simple,
            "method_name": method_name,
            "arg_count": arg_count,
        }

    def _deepcopy_jsonish(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return {k: self._deepcopy_jsonish(v) if isinstance(v, dict) else self._deepcopy_list(v) for k, v in obj.items()}
        return obj

    def _deepcopy_list(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._deepcopy_jsonish(x) if isinstance(x, dict) else self._deepcopy_list(x) for x in value]
        return value
