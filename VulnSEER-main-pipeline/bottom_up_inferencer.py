import json
import subprocess
import os
import re
import time

try:
    from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIError
except ModuleNotFoundError:
    OpenAI = None
    APIConnectionError = APITimeoutError = RateLimitError = APIError = Exception


# ==========================================
# Framework stage: sink requirement and layer-wise exploit-goal inference
# ==========================================

class LayerWiseExploitGoalInferencer:
    # def __init__(self, cpg_path, joern_bin, model="deepseek-v3.2", debug_prompt_dir="./output/debug_prompts"):
    #     self.cpg_path = cpg_path
    #     self.joern_bin = joern_bin
    #     self.model = model
    #     self.debug_prompt_dir = debug_prompt_dir
    #     os.makedirs(self.debug_prompt_dir, exist_ok=True)

    #     api_key = os.environ.get("OPENAI_API_KEY")
    #     if not api_key:
    #         print("[-] Warning: OPENAI_API_KEY environment variable was not found.")

    #     self.client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")
    #     self._method_context_cache = {}
    def __init__(
        self,
        cpg_path,
        joern_bin,
        model="qwen/qwen3.5-plus-02-15",
        debug_prompt_dir="./output/debug_prompts",
        base_url=None,
    ):
        self.cpg_path = cpg_path
        self.joern_bin = joern_bin
        self.model = model
        self.debug_prompt_dir = debug_prompt_dir
        os.makedirs(self.debug_prompt_dir, exist_ok=True)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[-] Warning: OPENAI_API_KEY environment variable was not found.")

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"

        print(f"[DEBUG] LLM model    = {self.model}")
        print(f"[DEBUG] LLM base_url = {self.base_url}")

        if OpenAI is None:
            raise ImportError("Missing openai Python package; install it first with: pip install openai")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers={
                "X-OpenRouter-Title": "VulnSEER-main-pipeline",
            },
        )

        self._method_context_cache = {}
    # def __init__(self, cpg_path, joern_bin, model="openai/gpt-5.2", debug_prompt_dir="./output/debug_prompts"):
    #     self.cpg_path = cpg_path
    #     self.joern_bin = joern_bin
    #     self.model = model
    #     self.debug_prompt_dir = debug_prompt_dir
    #     os.makedirs(self.debug_prompt_dir, exist_ok=True)

    #     api_key = os.environ.get("OPENAI_API_KEY")
    #     if not api_key:
    #         print("[-] Warning: OPENAI_API_KEY environment variable was not found.")

    #     self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    #     self._method_context_cache = {}

    def set_cpg_path(self, cpg_path: str):
        self.cpg_path = cpg_path

    def parse_legacy_helper_text(self, raw_text: str):
        items = []
        if not raw_text:
            return items

        blocks = [b.strip() for b in raw_text.split('---') if b.strip()]

        for idx, block in enumerate(blocks):
            lines = [x.rstrip() for x in block.splitlines() if x.strip()]
            if not lines:
                continue

            signature = "unknown"
            content_lines = []

            if lines[0].startswith("Method:"):
                signature = lines[0].replace("Method:", "").strip()

            impl_started = False
            for line in lines[1:]:
                if line.startswith("Implementation:"):
                    impl_started = True
                    after = line.replace("Implementation:", "").strip()
                    if after:
                        content_lines.append(after)
                    continue
                if impl_started:
                    content_lines.append(line)

            content = "\n".join(content_lines).strip() if content_lines else "<empty>"

            if signature != "unknown":
                helper_name = signature.split(":")[0].split(".")[-1]
            else:
                helper_name = f"helper_{idx}"

            items.append({
                "item_id": f"helper_fallback_{idx}",
                "source_type": "helper",
                "name": helper_name,
                "signature": signature,
                "content": content
            })

        return items

    def parse_legacy_class_context_text(self, raw_text: str):
        items = []
        if not raw_text:
            return items

        blocks = [b.strip() for b in raw_text.split('---') if b.strip()]
        field_idx = 0
        ctor_idx = 0

        for block in blocks:
            lines = [l.rstrip() for l in block.splitlines() if l.strip()]
            if not lines:
                continue

            if lines[0].startswith("Constructor:"):
                signature = lines[0].replace("Constructor:", "").strip()

                content_lines = []
                impl_started = False
                for line in lines[1:]:
                    if line.startswith("Implementation:"):
                        impl_started = True
                        after = line.replace("Implementation:", "").strip()
                        if after:
                            content_lines.append(after)
                        continue
                    if impl_started:
                        content_lines.append(line)

                content = "\n".join(content_lines).strip() if content_lines else "<empty>"

                items.append({
                    "item_id": f"ctor_fallback_{ctor_idx}",
                    "source_type": "constructor",
                    "name": "<init>",
                    "signature": signature if signature else "<init>",
                    "content": content
                })
                ctor_idx += 1
                continue

            for line in lines:
                if line.startswith("Field:"):
                    content = line.replace("Field:", "").strip()
                    source_type = "constant" if "static final" in content else "field"
                    clean = content.replace(";", "").strip()
                    tokens = clean.split()
                    name = tokens[-1] if tokens else f"field_{field_idx}"

                    items.append({
                        "item_id": f"{source_type}_fallback_{field_idx}",
                        "source_type": source_type,
                        "name": name,
                        "signature": name,
                        "content": content
                    })
                    field_idx += 1

        return items

    def normalize_candidate_items(self, method_info: dict):
        if method_info.get("is_external", False):
            print("[DEBUG:normalize] external method -> skip local helper/class-context normalization")
            return {"helpers": [], "class_context": []}

        helpers = method_info.get("helpers", []) or []
        class_context = method_info.get("class_context", []) or []

        print(f"[DEBUG:normalize] before cleanup helpers={len(helpers)}, class_context={len(class_context)}")

        def is_meaningful_item(item):
            source_type = (item.get("source_type") or "").strip().lower()
            content = (item.get("content") or "").strip()
            signature = (item.get("signature") or "").strip()
            name = (item.get("name") or "").strip()

            if not content:
                return False

            lowered_content = content.lower()
            lowered_sig = signature.lower()
            lowered_name = name.lower()

            if lowered_content in {"<empty>", "unknown", "none", "null"}:
                return False

            noisy_keywords = [
                "<operator>", "logger", "log.", "println", "printstacktrace", "tostring", "hashcode", "equals("
            ]
            if any(k in lowered_sig for k in noisy_keywords):
                return False
            if any(k in lowered_content for k in ["logger", "log.", "println", "printstacktrace"]):
                return False

            if lowered_name.startswith("get") or lowered_name.startswith("set"):
                return False

            if lowered_name == "<init>" and lowered_content == "<empty>":
                return False

            if source_type == "constructor":
                normalized = " ".join(lowered_content.split())
                if (
                    "specialinvoke this.<java.lang.object: void <init>()>()" in normalized
                    and "return;" in normalized
                    and normalized.count("specialinvoke") == 1
                ):
                    constructor_signal_keywords = [
                        " = ", "put(", "add(", "config", "handler", "factory", "session", "request",
                        "response", "token", "password", "encoder", "decoder"
                    ]
                    if not any(k in lowered_content for k in constructor_signal_keywords):
                        return False

            if source_type in {"field", "constant"}:
                low_value_field_keywords = ["serialversionuid", "logger", "log"]
                if lowered_name in low_value_field_keywords:
                    return False

            return True

        def dedup_items(items):
            seen = set()
            deduped = []
            for item in items:
                key = (item.get("source_type", ""), item.get("signature", ""), item.get("content", ""))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            return deduped

        def normalize_source_type(item):
            source_type = (item.get("source_type") or "").strip().lower()
            if source_type.startswith("helper"):
                item["source_type"] = "helper"
            elif source_type.startswith("ctor") or source_type == "constructor_fallback":
                item["source_type"] = "constructor"
            elif source_type.startswith("field"):
                item["source_type"] = "field"
            elif source_type.startswith("constant"):
                item["source_type"] = "constant"
            return item

        helpers = [normalize_source_type(dict(x)) for x in helpers]
        class_context = [normalize_source_type(dict(x)) for x in class_context]

        helpers = [x for x in helpers if is_meaningful_item(x)]
        class_context = [x for x in class_context if is_meaningful_item(x)]

        helpers = dedup_items(helpers)
        class_context = dedup_items(class_context)

        print(f"[DEBUG:normalize] after structured cleanup helpers={len(helpers)}, class_context={len(class_context)}")

        raw_helpers_text = method_info.get("raw_helpers_text", "") or ""
        raw_class_context_text = method_info.get("raw_class_context_text", "") or ""

        if not helpers and raw_helpers_text.strip():
            print("[DEBUG:normalize] structured helpers empty, fallback to raw_helpers_text")
            fallback_helpers = self.parse_legacy_helper_text(raw_helpers_text)
            fallback_helpers = [normalize_source_type(dict(x)) for x in fallback_helpers]
            fallback_helpers = [x for x in fallback_helpers if is_meaningful_item(x)]
            helpers = dedup_items(fallback_helpers)

        if not class_context and raw_class_context_text.strip():
            print("[DEBUG:normalize] structured class_context empty, fallback to raw_class_context_text")
            fallback_class = self.parse_legacy_class_context_text(raw_class_context_text)
            fallback_class = [normalize_source_type(dict(x)) for x in fallback_class]
            fallback_class = [x for x in fallback_class if is_meaningful_item(x)]
            class_context = dedup_items(fallback_class)

        final_helpers = []
        final_class_context = []

        for item in helpers:
            st = item.get("source_type", "")
            if st == "helper":
                final_helpers.append(item)

        for item in class_context:
            st = item.get("source_type", "")
            if st in {"field", "constant", "constructor"}:
                final_class_context.append(item)

        for item in helpers:
            st = item.get("source_type", "")
            if st not in {"helper", "field", "constant", "constructor"}:
                sig = (item.get("signature") or "").lower()
                name = (item.get("name") or "").lower()
                if "<init>" in sig or name == "<init>":
                    item["source_type"] = "constructor"
                    final_class_context.append(item)
                else:
                    item["source_type"] = "helper"
                    final_helpers.append(item)

        final_helpers = final_helpers[:8]
        final_class_context = final_class_context[:8]

        print(f"[DEBUG:normalize] after fallback helpers={len(final_helpers)}, class_context={len(final_class_context)}")
        if final_helpers:
            print(f"[DEBUG:normalize] first helper={final_helpers[0]}")
        if final_class_context:
            print(f"[DEBUG:normalize] first class_context={final_class_context[0]}")

        for item in final_helpers:
            item["priority_hint"] = "medium"
        for item in final_class_context:
            if item.get("source_type") in {"field", "constant", "constructor"}:
                item["priority_hint"] = "high"
            else:
                item["priority_hint"] = "medium"

        return {"helpers": final_helpers, "class_context": final_class_context}

    def _render_context_items(self, items):
        if not items:
            return "(None)"
        lines = []
        for item in items:
            lines.append(
                f"[{item.get('source_type','unknown')}] {item.get('name','unknown')} | "
                f"{item.get('signature','unknown')}\n{item.get('content','')}"
            )
        return "\n\n".join(lines)

    def get_method_context_from_cpg(self, method_fullname, chain_context=None):
        cache_key = (self.cpg_path, method_fullname)
        if cache_key in self._method_context_cache:
            print(f"[DEBUG:get_method_context] cache hit: {method_fullname}")
            return json.loads(json.dumps(self._method_context_cache[cache_key]))

        safe_fullname = method_fullname.replace('"', '\\"')
        joern_commands = f'''
    importCpg("{self.cpg_path}")
    val mOption = cpg.method.fullNameExact("{safe_fullname}").headOption
    if (mOption.isDefined) {{
        val m = mOption.get
        val typeDecl = m.typeDecl.headOption

        println("===CODE_START===")
        val rawCode: String = {{
            val content = m.content.getOrElse("")
            if (content.nonEmpty) content else m.code
        }}
        println(rawCode)
        println("===CODE_END===")

        println("===PARAMS_START===")
        m.parameter.filterNot(_.name == "this").sortBy(_.order).foreach(p =>
            println(s"${{p.name}}:::${{p.typeFullName}}")
        )
        println("===PARAMS_END===")

        println("===HELPER_RAW_START===")
        m.call.callee.distinct
        .filter(h => !h.fullName.startsWith("<operator>"))
        .filterNot(h => h.name.startsWith("get") || h.name.startsWith("set"))
        .filterNot(h => h.name == "toString")
        .filterNot(h =>
            h.name.toLowerCase.contains("logger") ||
            h.name.toLowerCase.contains("print") ||
            h.fullName.toLowerCase.contains("logger") ||
            h.fullName.toLowerCase.contains("println") ||
            h.fullName.toLowerCase.contains("printstacktrace")
        )
        .take(20)
        .foreach {{ h =>
            val hCode = h.content.getOrElse(h.code)
            val snippet = hCode.split("\\n").take(25).mkString("\\n")
            println(s"Method: ${{h.fullName}}")
            println(s"Implementation:\\n${{snippet}}")
            println("---")
        }}
        println("===HELPER_RAW_END===")

        println("===CLASS_RAW_START===")
        if (typeDecl.isDefined) {{
            typeDecl.get.member.code.foreach(c => println(s"Field: ${{c}}"))
            typeDecl.get.method.nameExact("<init>").foreach {{ ctor =>
                val ctorCode = ctor.content.getOrElse(ctor.code)
                println(s"Constructor: ${{ctor.fullName}}")
                println(s"Implementation:\\n${{ctorCode}}")
                println("---")
            }}
        }}
        println("===CLASS_RAW_END===")
    }} else {{
        println("[-] Method Not Found")
    }}
    exit
    '''

        method_info = {
            "code": "",
            "params": [],
            "helpers": [],
            "class_context": [],
            "raw_helpers_text": "",
            "raw_class_context_text": "",
            "method_found": False,
            "is_external": False,
            "no_context_reason": ""
        }

        env = os.environ.copy()
        java_home = env.get("JAVA_HOME")
        if java_home:
            env["PATH"] = f"{java_home}/bin:" + env.get("PATH", "")

        try:
            result = subprocess.run(
                [self.joern_bin],
                input=joern_commands,
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            stdout = result.stdout
            stderr = result.stderr

            print(f"[DEBUG:get_method_context] method={method_fullname}")
            print(f"[DEBUG:get_method_context] returncode={result.returncode}")
            print(f"[DEBUG:get_method_context] stdout_len={len(stdout)}")
            print(f"[DEBUG:get_method_context] stderr_len={len(stderr)}")

            if stderr.strip():
                print("[DEBUG:get_method_context] stderr preview:")
                print(stderr[:1000])

            if stdout.strip():
                print("[DEBUG:get_method_context] stdout preview:")
                print(stdout[:2000])

            if "[-] Method Not Found" in stdout:
                print(f"[!] Warning: method not found in CPG: {method_fullname}")
                method_info["method_found"] = False
                method_info["is_external"] = True
                method_info["no_context_reason"] = "Method not found in current project CPG; likely external library method."
                self._method_context_cache[cache_key] = method_info
                return json.loads(json.dumps(method_info))

            def extract_section(start_tag, end_tag):
                start = stdout.find(start_tag)
                end = stdout.find(end_tag)
                if start != -1 and end != -1:
                    return stdout[start + len(start_tag):end].strip()
                return ""

            method_info["code"] = extract_section("===CODE_START===", "===CODE_END===")

            param_str = extract_section("===PARAMS_START===", "===PARAMS_END===")
            if param_str:
                for line in param_str.split("\n"):
                    line = line.strip()
                    if ":::" in line:
                        parts = line.split(":::", 1)
                        method_info["params"].append({
                            "name": parts[0].strip(),
                            "type": parts[1].strip()
                        })

            raw_helpers = extract_section("===HELPER_RAW_START===", "===HELPER_RAW_END===")
            raw_class = extract_section("===CLASS_RAW_START===", "===CLASS_RAW_END===")

            method_info["raw_helpers_text"] = raw_helpers
            method_info["raw_class_context_text"] = raw_class

            helper_blocks = [b.strip() for b in raw_helpers.split("---") if b.strip()]
            parsed_helpers = []
            helper_idx = 0

            for block in helper_blocks:
                lines = [x.rstrip() for x in block.splitlines() if x.strip()]
                if not lines:
                    continue

                signature = ""
                content_lines = []

                if lines[0].startswith("Method:"):
                    signature = lines[0].replace("Method:", "").strip()

                impl_started = False
                for line in lines[1:]:
                    if line.startswith("Implementation:"):
                        impl_started = True
                        after = line.replace("Implementation:", "").strip()
                        if after:
                            content_lines.append(after)
                        continue
                    if impl_started:
                        content_lines.append(line)

                content = "\n".join(content_lines).strip() if content_lines else "<empty>"
                lowered_sig = signature.lower()
                if any(k in lowered_sig for k in ["println", "logger", "log.", "tostring"]):
                    continue

                helper_name = signature.split(".")[-1].split(":")[0] if signature else f"helper_{helper_idx}"
                parsed_helpers.append({
                    "item_id": f"helper_{helper_idx}",
                    "source_type": "helper",
                    "name": helper_name,
                    "signature": signature,
                    "content": content
                })
                helper_idx += 1

            method_info["helpers"] = parsed_helpers

            class_items = []
            class_idx = 0
            class_blocks = [b.strip() for b in raw_class.split("---") if b.strip()]
            for block in class_blocks:
                lines = [x.rstrip() for x in block.splitlines() if x.strip()]
                if not lines:
                    continue

                if lines[0].startswith("Constructor:"):
                    signature = lines[0].replace("Constructor:", "").strip()
                    content_lines = []
                    impl_started = False
                    for line in lines[1:]:
                        if line.startswith("Implementation:"):
                            impl_started = True
                            after = line.replace("Implementation:", "").strip()
                            if after:
                                content_lines.append(after)
                            continue
                        if impl_started:
                            content_lines.append(line)

                    content = "\n".join(content_lines).strip() if content_lines else "<empty>"
                    class_items.append({
                        "item_id": f"ctor_{class_idx}",
                        "source_type": "constructor",
                        "name": "<init>",
                        "signature": signature,
                        "content": content
                    })
                    class_idx += 1
                    continue

                for line in lines:
                    if line.startswith("Field:"):
                        field_code = line.replace("Field:", "").strip()
                        source_type = "constant" if "static final" in field_code else "field"
                        tokens = field_code.replace(";", "").split()
                        field_name = tokens[-1] if tokens else f"field_{class_idx}"
                        class_items.append({
                            "item_id": f"{source_type}_{class_idx}",
                            "source_type": source_type,
                            "name": field_name,
                            "signature": f"{method_fullname}#{field_name}",
                            "content": field_code
                        })
                        class_idx += 1

            method_info["class_context"] = class_items
            method_info["method_found"] = True
            method_info["is_external"] = False
            method_info["no_context_reason"] = ""

            print(f"[DEBUG:structured] helpers={len(method_info['helpers'])}, class_context={len(method_info['class_context'])}")
            if method_info["helpers"]:
                print(f"[DEBUG:structured] first helper={method_info['helpers'][0]}")
            if method_info["class_context"]:
                print(f"[DEBUG:structured] first class_context={method_info['class_context'][0]}")

            self._method_context_cache[cache_key] = method_info
            return json.loads(json.dumps(method_info))

        except subprocess.TimeoutExpired:
            print(f"[-] Joern query timed out: {method_fullname}")
            return method_info
        except Exception as e:
            print(f"[-] Extraction failed: {e}")
            self._method_context_cache[cache_key] = method_info
            return json.loads(json.dumps(method_info))
    
    def _extract_llm_text(self, response):
        """
        Handle response shapes used by OpenAI, Claude, and common gateway APIs.
        """
        if response is None:
            return ""

        # OpenAI standard format.
        if hasattr(response, "choices"):
            try:
                return response.choices[0].message.content
            except Exception:
                pass

        # Some gateway APIs.
        if hasattr(response, "output_text"):
            return response.output_text

        # Common Claude-style format.
        if hasattr(response, "content"):
            try:
                if isinstance(response.content, list):
                    return "".join([c.text for c in response.content if hasattr(c, "text")])
            except Exception:
                pass

        return str(response)
    
    def _extract_json(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # Wrapped in ```json fences.
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Find the first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]

        return text

    def _parse_llm_json(self, raw):
        """
        Handle several common response forms:
        1. {"arg0": {...}}
        2. ```json {"arg0": {...}} ```
        3. ["{\"arg0\": {...}}"]
        4. "{\"arg0\": {...}}"
        """
        if raw is None:
            raise ValueError("Empty LLM response")

        if not isinstance(raw, str):
            raw = str(raw)

        raw = raw.strip()

        def normalize(obj):
            if isinstance(obj, dict):
                return obj

            if isinstance(obj, list):
                if len(obj) == 1:
                    return normalize(obj[0])
                raise ValueError(f"LLM returned a list with multiple elements: {obj}")

            if isinstance(obj, str):
                s = obj.strip()
                if not s:
                    raise ValueError("LLM returned empty string")

                # If the string itself is another JSON blob, keep decoding it.
                return normalize(json.loads(s))

            raise ValueError(f"Unsupported parsed JSON type: {type(obj)}")

        # 1. First try json.loads(raw) directly.
        try:
            return normalize(json.loads(raw))
        except Exception:
            pass

        # 2. Then try extracting JSON from markdown or plain text.
        json_text = self._extract_json(raw)

        try:
            return normalize(json.loads(json_text))
        except Exception:
            pass

        # 3. If the extracted text is escaped JSON such as {\"arg0\":...},
        #    decode it as a JSON string first, then parse the dict.
        try:
            unescaped = json.loads(f'"{json_text}"')
            return normalize(json.loads(unescaped))
        except Exception as e:
            raise e

    def generate_sink_requirement_from_exploit_sketch(self, exploit_sketch_dir, sink_method):
        print(f"\n[*] Searching and analyzing sink-level exploit sketches in: {exploit_sketch_dir}")
        if not os.path.exists(exploit_sketch_dir):
            raise FileNotFoundError(f"Directory does not exist: {exploit_sketch_dir}")

        java_files = [f for f in os.listdir(exploit_sketch_dir) if f.endswith(".java")]
        if not java_files:
            raise FileNotFoundError(f"No .java testcase files found in {exploit_sketch_dir}.")

        target_file = os.path.join(exploit_sketch_dir, java_files[0])
        print(f"[*] Selected exploit sketch file: {target_file}")

        with open(target_file, "r", encoding="utf-8") as f:
            testcase_code = f.read()

        prompt = f"""You are a senior vulnerability researcher.
Your task is to analyze a Java sink-level exploit sketch and extract the generalized sink requirement for the vulnerable sink method.

=== Sink Method ===
{sink_method}

=== Sink-Level Exploit Sketch Source Code ===
```java
{testcase_code}
```

=== Task Instructions ===
Analyze how the exploit sketch constructs the malicious input to trigger the vulnerability.
The provided sketch only represents ONE specific instance of the exploit. You must GENERALIZE the payload.
Abstract the core malicious characteristics required to trigger the vulnerability class rather than copying exact hardcoded strings.

=== OUTPUT FORMAT ===
You MUST return a SINGLE valid JSON object representing the constraints for the sink method's parameters.
Rules:
- Do NOT include any explanation
- Do NOT include markdown
- Do NOT include ```json
- Output must start with '{' and end with '}'
- Even if unsure, return a valid JSON
Failure to follow this will break the system.
The JSON schema MUST follow this exact format (do NOT add extra keys outside of arg0, arg1, etc.):
{{
  "arg0": {{
    "type": "[Fully qualified Java type, e.g., java.lang.String]",
    "placeholder": "<SINK_ARG0>",
    "description": "[Generalized description of the malicious payload characteristics based on the testcase]"
  }}
}}
"""
        print(f"[*] Requesting LLM generalization for the sink-level requirement...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a JSON-only vulnerability analyzer. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result_text = self._extract_llm_text(response)
        try:
            raw = self._extract_llm_text(response)

            print("[DEBUG] raw LLM:", str(raw)[:1000])

            sink_requirement = self._parse_llm_json(raw)

            print(f"[+] Sink-level requirement generalized successfully:\n{json.dumps(sink_requirement, indent=2, ensure_ascii=False)}")
            return sink_requirement

        except Exception as e:
            debug_path = os.path.join(self.debug_prompt_dir, "sink_requirement_raw.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(str(raw) if "raw" in locals() and raw else "<EMPTY>")

            print(f"[-] JSON parsing failed: {e}")
            print(f"[-] Raw output written to: {debug_path}")
            raise

    def build_prompt(self, caller, callee, method_info, callee_requirement, is_lowest_layer):
        params = method_info.get("params", [])
        param_str = f"The caller method fi has {len(params)} parameter(s).\n"
        for i, p in enumerate(params):
            param_str += f"  - arg{i}: {p['name']} ({p['type']})\n"

        req_str = json.dumps(callee_requirement, indent=2, ensure_ascii=False)

        sink_requirement_section = ""
        if is_lowest_layer:
            sink_requirement_section = f"""=== Sink-Level Requirement ===
This is the lowest layer (vulnerable sink).
An exploit sketch has been extracted. The vulnerable sink must receive the following arguments:
{req_str}
When you want fi-1 to receive exactly the same runtime object as in the exploit sketch,
you MUST use the placeholder string "<SINK_ARGi>" in candidate_inputs.args...
"""

        source_code = method_info.get("code") or "(Code not available)"
        helpers = self._render_context_items(method_info.get("helpers", []))
        class_context = self._render_context_items(method_info.get("class_context", []))

        if len(params) == 0:
            args_rule = '- Because the caller has 0 parameters, every "args" array MUST be exactly [].'
        else:
            args_rule = (
                f'- The "args" array in each element of "candidate_inputs" MUST have exactly '
                f'{len(params)} element(s), matching the caller\'s parameters in order.'
            )

        prompt = f"""You are a symbolic execution engine for Java.
You perform Hoare-logic style reasoning for a single step in a call chain.

=== Current Layer ===
Caller method (fi): {caller}
Callee method (fi-1): {callee}

=== Caller Parameters ===
{param_str}

=== Method Source Code ===
{source_code}

=== Helper Methods Source (Internal) ===
{helpers}

=== Class Context (Constructors & Constants) ===
{class_context}

=== Callee Invocation Requirement ===
To continue the execution down the chain, the callee (fi-1) must be invoked.
The callee requires the following conditions (from the layer below):
{req_str}

{sink_requirement_section}
=== Your Goals & Reasoning Process ===
1. Analyze the helper methods and class context above.
2. Infer caller-level preconditions such that fi will invoke fi-1 with arguments satisfying the above constraints.
3. If a field is an interface, identify a concrete implementation and construct its structure in JSON.

=== OUTPUT FORMAT ===
You MUST return a SINGLE valid JSON object:
{{
  "caller_precondition": ["...", "..."],
  "candidate_inputs": [
    {{"args": ["arg0_value", "arg1_value"]}}
  ]
}}

=== Constraints ===
- The output MUST be valid JSON.
- Do NOT use code expressions such as constructors or method calls.
- Each argument in "args" MUST be a plain JSON literal.
- For object/reference-type arguments whose exact internal state is not important, use a Java-runtime-style instance string such as "java.lang.Object@16e48ada" instead of code expressions like "new java.lang.Object()".
- Do NOT use structured typed placeholder objects such as {{"@type": "java.lang.Object", "@constraint": "any_non_null_instance"}}.
- If the value only needs to be a non-null instance of some Java reference type, prefer a plausible concrete runtime-style string matching that type, for example:
  - "java.lang.Object@16e48ada"
  - "java.util.HashMap@1a2b3c4d"
  - "com.example.MyBean@4f3f5b24"
{args_rule}
- If some constraints involve internal fields or configuration of fi, express them ONLY in "caller_precondition".
- If you need to reuse the exact sink argument object, use "<SINK_ARG0>", "<SINK_ARG1>", etc.
"""
        return prompt

    def safe_chat_completion(
        self,
        messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_retries=4,
        retry_base_sleep=2.0,
        fallback_json=None
    ):
        if fallback_json is None:
            fallback_json = {"status": "resolved", "arg_values": {}, "field_values": {}}

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature
                )
                raw = self._extract_llm_text(response)

                print(f"[DEBUG] raw LLM (attempt={attempt}/{max_retries}): {str(raw)[:1000] if raw else '<EMPTY RESPONSE>'}")

                return self._parse_llm_json(raw)
            except json.JSONDecodeError as e:
                print(f"[-] LLM returned invalid JSON (attempt={attempt}/{max_retries}): {e}")
                last_err = e
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
                print(f"[-] LLM call failed (attempt={attempt}/{max_retries}): {type(e).__name__}: {e}")
                last_err = e
            except Exception as e:
                print(f"[-] Unexpected LLM exception (attempt={attempt}/{max_retries}): {type(e).__name__}: {e}")
                last_err = e
            except json.JSONDecodeError as e:
                raw_preview = raw if 'raw' in locals() else "<NO RAW>"
                debug_path = os.path.join(self.debug_prompt_dir, f"llm_invalid_json_attempt_{attempt}.txt")
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(raw_preview)

                print(f"[-] LLM returned invalid JSON (attempt={attempt}/{max_retries}): {e}")
                print(f"[-] Raw response saved to: {debug_path}")
                last_err = e

            if attempt < max_retries:
                sleep_sec = retry_base_sleep * attempt
                print(f"[*] Retrying after {sleep_sec:.1f}s...")
                time.sleep(sleep_sec)

        print(f"[!] LLM failed repeatedly; returning fallback result: {fallback_json}")
        return fallback_json

    def call_llm(self, prompt):
        print(f"[*] Requesting LLM ({self.model}) to infer current-layer constraints...")
        return self.safe_chat_completion(
            messages=[
                {"role": "system", "content": "You are a JSON-only symbolic execution engine. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_retries=4,
            retry_base_sleep=2.0,
            fallback_json={"error": "llm_failed", "caller_precondition": [], "candidate_inputs": []}
        )

    def _coerce_path_context_pair(self, chain_or_pair):
        if isinstance(chain_or_pair, dict) and "rho" in chain_or_pair and "I_rho" in chain_or_pair:
            return chain_or_pair["rho"], chain_or_pair.get("I_rho", {})
        return chain_or_pair, {}

    def _get_indexed_routine_context(self, path_context_index, index, method_fullname):
        if not path_context_index:
            return None

        routine_contexts = path_context_index.get("routine_contexts", {}) or {}
        context = routine_contexts.get(f"r{index}")
        if context is None:
            for item in routine_contexts.values():
                if item.get("method") == method_fullname or item.get("method_signature") == method_fullname:
                    context = item
                    break

        if context is None:
            return None

        return self._routine_context_to_method_info(context)

    def _routine_context_to_method_info(self, context):
        method_info = dict(context)
        method_info["code"] = (
            context.get("source_code")
            or context.get("code")
            or context.get("pruned_source_code")
            or context.get("method_body")
            or ""
        )
        method_info["params"] = (
            context.get("parameters")
            or context.get("formal_parameters")
            or context.get("params")
            or []
        )
        method_info["helpers"] = self._path_context_helpers_to_items(context)
        method_info["class_context"] = self._path_context_class_context_to_items(context)
        method_info["method_found"] = context.get("method_found", bool(method_info["code"]))
        method_info["is_external"] = context.get("is_external", False)
        method_info["no_context_reason"] = context.get("missing_reason") or context.get("no_context_reason", "")
        method_info["_from_path_context"] = True
        return method_info

    def _path_context_helpers_to_items(self, context):
        if context.get("helpers"):
            return context.get("helpers") or []

        items = []
        for idx, helper in enumerate(context.get("local_helpers", []) or []):
            items.append({
                "item_id": f"helper_{idx}",
                "source_type": "helper",
                "name": helper.get("name", ""),
                "signature": helper.get("signature", ""),
                "content": helper.get("source_code", ""),
            })
        return items

    def _path_context_class_context_to_items(self, context):
        if context.get("class_context"):
            return context.get("class_context") or []

        items = []
        for idx, field in enumerate(context.get("field_accesses", []) or []):
            items.append({
                "item_id": f"field_access_{idx}",
                "source_type": "field",
                "name": field.get("name", ""),
                "signature": field.get("signature", ""),
                "content": field.get("code", ""),
            })
        for idx, field in enumerate(context.get("constants", []) or []):
            items.append({
                "item_id": f"constant_{idx}",
                "source_type": "constant",
                "name": field.get("name", ""),
                "signature": field.get("signature", ""),
                "content": field.get("code", ""),
            })
        for idx, ctor in enumerate(context.get("constructors", []) or []):
            items.append({
                "item_id": f"constructor_{idx}",
                "source_type": "constructor",
                "name": "<init>",
                "signature": ctor.get("signature", ""),
                "content": ctor.get("source_code", ""),
            })
        hierarchy = context.get("class_hierarchy", {}) or {}
        if hierarchy:
            items.append({
                "item_id": "class_hierarchy",
                "source_type": "class_hierarchy",
                "name": context.get("declaring_type", ""),
                "signature": context.get("declaring_type", ""),
                "content": json.dumps(hierarchy, ensure_ascii=False),
            })
        for idx, obj in enumerate(context.get("object_definitions", []) or []):
            items.append({
                "item_id": f"object_definition_{idx}",
                "source_type": "object_definition",
                "name": obj.get("type", ""),
                "signature": context.get("method", ""),
                "content": obj.get("code", ""),
            })
        return items

    def infer_layerwise_exploit_goals(self, chain_or_pair, sink_requirement):
        chain, path_context_index = self._coerce_path_context_pair(chain_or_pair)
        print(f"\n[+] Starting layer-wise backward exploit-goal inference (Chain length: {len(chain)})")
        exploit_goal_map = {}
        sink_method = chain[-1]
        current_requirement = sink_requirement
        exploit_goal_map[sink_method] = current_requirement

        debug_dir = self.debug_prompt_dir
        os.makedirs(debug_dir, exist_ok=True)

        for i in range(len(chain) - 2, -1, -1):
            step_num = len(chain) - 1 - i
            caller = chain[i]
            callee = chain[i + 1]
            is_lowest_layer = (i == len(chain) - 2)

            print(f"\n--- Layer analysis (Step {step_num}): {caller} -> {callee} ---")
            method_info = self._get_indexed_routine_context(path_context_index, i, caller)
            if method_info is not None:
                print(f"[*] Using I_rho routine context: r{i}")
            else:
                method_info = self.get_method_context_from_cpg(caller)
            if not method_info.get("code"):
                print(f"[!] Warning: source code for {caller} was not found in the CPG.")

            prompt = self.build_prompt(caller, callee, method_info, current_requirement, is_lowest_layer)
            safe_caller_name = re.sub(r'[^a-zA-Z0-9_\.]', '_', caller.split(':')[0].replace('.', '_'))
            debug_file_path = os.path.join(debug_dir, f"exploit_goal_step_{step_num}_{safe_caller_name}_prompt.txt")
            with open(debug_file_path, "w", encoding="utf-8") as df:
                df.write(prompt)
            print(f"[*] Prompt saved to: {debug_file_path}")

            llm_result = self.call_llm(prompt)
            print(f"[+] Inference result:\n{json.dumps(llm_result, indent=2, ensure_ascii=False)}")

            exploit_goal_map[caller] = llm_result
            current_requirement = llm_result

        return exploit_goal_map

# ==========================================
# Framework stage: execution-plan construction
# ==========================================

class ExecutionPlanBuilder:
    @staticmethod
    def build_execution_plan(chain, exploit_goal_map, path_context_pair=None):
        print("\n[+] Building entry-to-sink execution plan...")
        execution_plan = []

        for i, method in enumerate(chain):
            layer_info = {
                "step_index": i,
                "method_signature": method,
                "routine_context_id": f"r{i}",
            }

            if i == 0:
                layer_info["layer_type"] = "ENTRY"
            elif i == len(chain) - 1:
                layer_info["layer_type"] = "SINK"
            else:
                layer_info["layer_type"] = "INTERMEDIATE"

            layer_info["calls_next"] = chain[i + 1] if i < len(chain) - 1 else None
            layer_info["transition_context_id"] = f"c{i}" if i < len(chain) - 1 else None
            goal = exploit_goal_map.get(method, {})

            if layer_info["layer_type"] == "SINK":
                layer_info["sink_requirements"] = goal
            else:
                layer_info["preconditions"] = goal.get("caller_precondition", [])
                layer_info["candidate_inputs"] = goal.get("candidate_inputs", [])

            execution_plan.append(layer_info)

        print(f"[*] Execution plan assembled successfully with {len(execution_plan)} plan units.")
        return execution_plan

# ==========================================
# Framework stage: LLM-guided exploit-state resolution
# ==========================================

class ExploitStateResolver:
    def __init__(self, inferencer):
        self.inferencer = inferencer
        self.client = inferencer.client
        self.model = inferencer.model

    def set_cpg_path(self, cpg_path: str):
        self.inferencer.set_cpg_path(cpg_path)

    def _get_layer_routine_context(self, layer_idx, current_method, layer, path_context_pair=None):
        if path_context_pair:
            path_context_index = path_context_pair.get("I_rho", {})
            context_id = layer.get("routine_context_id") or f"r{layer_idx}"
            routine_contexts = path_context_index.get("routine_contexts", {}) or {}
            context = routine_contexts.get(context_id)
            if context is None:
                context = self.inferencer._get_indexed_routine_context(path_context_index, layer_idx, current_method)
            if context is not None:
                if context.get("_from_path_context"):
                    return context
                return self.inferencer._routine_context_to_method_info(context)

        return self.inferencer.get_method_context_from_cpg(current_method)

    def _get_layer_transition_context(self, layer_idx, layer, path_context_pair=None):
        if not path_context_pair:
            return None
        path_context_index = path_context_pair.get("I_rho", {})
        transition_contexts = path_context_index.get("transition_contexts", {}) or {}
        context_id = layer.get("transition_context_id") or f"c{layer_idx}"
        return transition_contexts.get(context_id)

    def _make_materialized_candidate(self, item_id, source_type, name, signature, content, context_ref):
        return {
            "item_id": item_id,
            "source_type": source_type,
            "name": name,
            "signature": signature,
            "content": content,
            "context_ref": context_ref,
            "materialized": True,
        }

    def _build_candidate_items_from_path_context(self, method_info, transition_context, layer_idx):
        routine_ref = f"r{layer_idx}"
        transition_ref = f"c{layer_idx}"
        candidates = []

        for idx, helper in enumerate(method_info.get("helpers", []) or []):
            candidates.append(self._make_materialized_candidate(
                f"{routine_ref}_helper_{idx}",
                helper.get("source_type", "helper"),
                helper.get("name", ""),
                helper.get("signature", ""),
                helper.get("content", ""),
                routine_ref,
            ))

        for idx, item in enumerate(method_info.get("class_context", []) or []):
            candidates.append(self._make_materialized_candidate(
                f"{routine_ref}_class_{idx}",
                item.get("source_type", "class_context"),
                item.get("name", ""),
                item.get("signature", ""),
                item.get("content", ""),
                routine_ref,
            ))

        if not transition_context:
            return candidates

        transition_sig = f"{transition_context.get('caller', '')} -> {transition_context.get('callee', '')}"
        call_site = transition_context.get("call_site", "")
        if call_site:
            content = "\n".join([
                f"Call site: {call_site}",
                f"Receiver expression: {transition_context.get('receiver_expression', '')}",
                "Actual arguments: " + ", ".join(transition_context.get("actual_arguments", [])),
            ]).strip()
            candidates.append(self._make_materialized_candidate(
                f"{transition_ref}_callsite",
                "transition_callsite",
                "call_site",
                transition_sig,
                content,
                transition_ref,
            ))

        if transition_context.get("nearby_slice"):
            candidates.append(self._make_materialized_candidate(
                f"{transition_ref}_slice",
                "transition_slice",
                "nearby_slice",
                transition_sig,
                transition_context.get("nearby_slice", ""),
                transition_ref,
            ))

        if transition_context.get("dynamic_dispatch"):
            candidates.append(self._make_materialized_candidate(
                f"{transition_ref}_dispatch",
                "dynamic_dispatch",
                "dynamic_dispatch",
                transition_sig,
                json.dumps(transition_context.get("dynamic_dispatch", []), ensure_ascii=False),
                transition_ref,
            ))

        if transition_context.get("propagation_context"):
            candidates.append(self._make_materialized_candidate(
                f"{transition_ref}_propagation",
                "propagation_context",
                "propagation_context",
                transition_sig,
                "\n".join(transition_context.get("propagation_context", [])),
                transition_ref,
            ))

        return candidates

    def _sanitize_arg_values(self, arg_values, params):
        if not isinstance(arg_values, dict):
            return {}

        valid_keys = {f"arg{i}" for i in range(len(params))}
        sanitized = {}

        for k, v in arg_values.items():
            if k in valid_keys:
                sanitized[k] = v
            else:
                print(f"[DEBUG:sanitize_arg_values] drop illegal arg key: {k}")

        return sanitized

    def _extract_illegal_arg_values_as_fields(self, arg_values, params):
        if not isinstance(arg_values, dict):
            return {}

        valid_keys = {f"arg{i}" for i in range(len(params))}
        migrated = {}

        for k, v in arg_values.items():
            if k not in valid_keys:
                migrated[f"migrated_from_{k}"] = v

        return migrated

    def _build_dynamic_arg_values_schema(self, params):
        if not params:
            return "{}"

        parts = []
        for i in range(len(params)):
            parts.append(f'"arg{i}": "<CONCRETE value>"')
        return "{ " + ", ".join(parts) + " }"
    
    def _dump_resolution_prompt(self, layer_idx, current_method, prompt):
        """
        Save the exploit-state resolution prompt into debug_prompt_dir.
        """

        debug_dir = self.inferencer.debug_prompt_dir
        os.makedirs(debug_dir, exist_ok=True)

        safe_method_name = re.sub(
            r'[^a-zA-Z0-9_\.]',
            '_',
            current_method.split(':')[0].split('.')[-2] + '.' + current_method.split(':')[0].split('.')[-1]
            if '.' in current_method.split(':')[0] else current_method.split(':')[0]
        )

        file_path = os.path.join(
            debug_dir,
            f"resolve_state_layer_{layer_idx}_{safe_method_name}_prompt.txt"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"[*] Exploit-state resolution prompt saved to: {file_path}")

    def build_exploit_state_resolution_prompt(
        self,
        layer_idx,
        current_method,
        next_method,
        method_info,
        layer_goal,
        sink_requirement,
        previous_layers_results
    ):
        params = method_info.get("params", [])
        param_str = "\n".join([f"  - arg{i}: {p['name']} ({p['type']})" for i, p in enumerate(params)]) or "(No parameters)"
        source_code = method_info.get("code") or "(Code not available)"
        helpers = method_info.get("helpers") or "(No selected helper methods)"
        class_context = method_info.get("class_context") or "(No selected class context)"

        goal_str = json.dumps(layer_goal, indent=2, ensure_ascii=False)
        sink_requirement_str = json.dumps(sink_requirement, indent=2, ensure_ascii=False)
        prev_states_str = json.dumps(previous_layers_results, indent=2, ensure_ascii=False) if previous_layers_results else "(None)"

        schema_description = ""
        if layer_idx == 0:
            schema_description = '  "description": "<A detailed natural language explanation of how this entry payload triggers the vulnerability based on the sink requirement>",\n'

        arg_values_schema = self._build_dynamic_arg_values_schema(params)

        if len(params) == 0:
            arg_constraints = """7. Because the current method has 0 parameters, "arg_values" MUST be {}.
8. Do NOT invent arg0/arg1/... keys that do not exist in the current method signature.
9. If a downstream call requires additional values (for example property names such as "class"), put them in "field_values" when they belong to internal selector/configuration state, not in "arg_values"."""
        else:
            allowed_keys = ", ".join([f'"arg{i}"' for i in range(len(params))])
            arg_constraints = f"""7. "arg_values" may ONLY contain these keys: {allowed_keys}.
8. Do NOT invent extra arg keys such as arg{len(params)} or beyond.
9. If a downstream call requires additional values (for example property names such as "class"), put them in "field_values" when they belong to internal selector/configuration state, not in "arg_values".
10. When a parameter can be ANY value that satisfies the required Java type and basic semantic constraints, do NOT invent fake Java code expressions such as:
   - "new java.lang.Object()"
   - "new HashMap<>()"
   - "someMethodCall()"
11. In such cases, prefer a Java-runtime-style concrete instance string instead of code, for example:
   - "java.lang.Object@16e48ada"
   - "java.util.HashMap@1a2b3c4d"
12. The value should look like a plausible runtime object instance identifier, not source code and not a typed JSON schema object.
13. Use this runtime-style object string only when the exact internal state of the object is not important for triggering the sink. If the exploit depends on specific internal fields, express those requirements in "field_values" instead."""

        prompt = f"""You are a symbolic execution engine for Java.
We are analyzing LAYER INDEX {layer_idx} of the call chain.
Method signature: {current_method}
Next method: {next_method}

=== Parameters ===
{param_str}

=== Method Source Code ===
{source_code}

=== Helper Methods Source (Selected by Selector) ===
{helpers}

=== Class Context (Selected by Selector) ===
{class_context}

=== Sink-Level Requirement ===
{sink_requirement_str}

=== Execution State from Previous (Higher) Layers ===
{prev_states_str}

=== Layer Exploit Goal ===
{goal_str}

=== Critical Guidelines ===
1. If you have enough information, you MUST generate concrete payload values. Do NOT emit placeholders like <SINK_ARG0> in the final answer.
2. Never use Java code expressions, constructors, or method calls in arg_values or field_values.
3. Maintain consistency with previous layer states.
4. If a value is still abstract, output the most concrete malicious value you can infer from sink-level constraints and local semantics.
5. For object/reference-type parameters whose exact internal content is not important, prefer a Java runtime object-instance style string such as "java.lang.Object@16e48ada" rather than code expressions or abstract schema objects.
6. Do NOT output structured typed placeholder objects such as:
   - {{"@type": "java.lang.Object", "@constraint": "any_non_null_instance"}}
{arg_constraints}

=== Output JSON Schema ===
Output one JSON object:
{{
  "status": "resolved",
{schema_description}  "arg_values": {arg_values_schema},
  "field_values": {{ "fieldName1": "<concrete value>" }}
}}
"""
        return prompt

    def resolve_execution_plan(self, execution_plan, path_context_pair=None):
        if not execution_plan or execution_plan[-1].get("layer_type") != "SINK":
            raise ValueError("Invalid execution_plan: the last layer must be SINK")

        sink_requirement = execution_plan[-1].get("sink_requirements", {})
        if not sink_requirement:
            raise ValueError("execution_plan is missing sink_requirements")

        final_payload = {"arg_values": {}, "field_values": {}}
        previous_layers_results = []
        carry_progress_delta = 0

        # Process ENTRY through the penultimate layer only; SINK is not solved again.
        for i, layer in enumerate(execution_plan[:-1]):
            current_method = layer["method_signature"]
            next_method = layer.get("calls_next") or "Sink"
            layer_goal = {
                "preconditions": layer.get("preconditions", []),
                "candidate_inputs": layer.get("candidate_inputs", [])
            }

            method_info = self._get_layer_routine_context(
                layer_idx=i,
                current_method=current_method,
                layer=layer,
                path_context_pair=path_context_pair,
            )
            current_params = method_info.get("params", []) or []
            transition_context = self._get_layer_transition_context(
                layer_idx=i,
                layer=layer,
                path_context_pair=path_context_pair,
            )

            if method_info.get("is_external", False):
                print(f"[DEBUG] Layer {i} | external method detected: {current_method}")
                filtered_method_info = {
                    "code": "(External library method implementation not available in current project CPG)",
                    "params": current_params,
                    "helpers": "(No helper methods available for external library method)",
                    "class_context": f"(No class context available: {method_info.get('no_context_reason', 'external method')})"
                }

                prompt = self.build_exploit_state_resolution_prompt(
                    i, current_method, next_method, filtered_method_info, layer_goal, sink_requirement, previous_layers_results
                )
                self._dump_resolution_prompt(i, current_method, prompt)
                print(f"[*] Layer {i} | Requesting LLM reasoning...")
                result = self.inferencer.safe_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a JSON-only symbolic engine."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_retries=4,
                    retry_base_sleep=2.0,
                    fallback_json={"status": "resolved", "arg_values": {}, "field_values": {}}
                )

                if result.get("status") == "resolved":
                    sanitized_arg_values = self._sanitize_arg_values(result.get("arg_values", {}), current_params)
                    migrated_illegal_args = self._extract_illegal_arg_values_as_fields(result.get("arg_values", {}), current_params)

                    if "field_values" not in result or not isinstance(result["field_values"], dict):
                        result["field_values"] = {}
                    result["field_values"].update(migrated_illegal_args)
                    result["arg_values"] = sanitized_arg_values

                    previous_layers_results.append({
                        "layer_index": i,
                        "method": current_method,
                        "inferred_state": result
                    })
                    if "field_values" in result and isinstance(result["field_values"], dict):
                        final_payload["field_values"].update(result["field_values"])
                    if i == 0:
                        if "arg_values" in result and isinstance(result["arg_values"], dict):
                            final_payload["arg_values"].update(result["arg_values"])
                        if "description" in result:
                            final_payload["description"] = result["description"]
                    carry_progress_delta = len(result.get("field_values", {})) + len(result.get("arg_values", {}))
                else:
                    carry_progress_delta = 0
                continue

            if method_info.get("_from_path_context"):
                candidate_items = self._build_candidate_items_from_path_context(
                    method_info=method_info,
                    transition_context=transition_context,
                    layer_idx=i,
                )
            elif method_info.get("candidate_items"):
                candidate_items = list(method_info.get("candidate_items", []))
            else:
                normalized = self.inferencer.normalize_candidate_items(method_info)
                candidate_items = normalized["helpers"] + normalized["class_context"]

            if not candidate_items:
                print(f"[DEBUG] Layer {i} | project-local method but no candidate context items.")
                filtered_method_info = {
                    "code": method_info.get("code", ""),
                    "params": current_params,
                    "helpers": "(No helper methods extracted)",
                    "class_context": "(No class context extracted)"
                }
                prompt = self.build_exploit_state_resolution_prompt(
                    i, current_method, next_method, filtered_method_info, layer_goal, sink_requirement, previous_layers_results
                )
                self._dump_resolution_prompt(i, current_method, prompt)
                print(f"[*] Layer {i} | Requesting LLM reasoning...")
                result = self.inferencer.safe_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a JSON-only symbolic engine."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_retries=4,
                    retry_base_sleep=2.0,
                    fallback_json={"status": "resolved", "arg_values": {}, "field_values": {}}
                )
                if result.get("status") == "resolved":
                    sanitized_arg_values = self._sanitize_arg_values(result.get("arg_values", {}), current_params)
                    migrated_illegal_args = self._extract_illegal_arg_values_as_fields(result.get("arg_values", {}), current_params)

                    if "field_values" not in result or not isinstance(result["field_values"], dict):
                        result["field_values"] = {}
                    result["field_values"].update(migrated_illegal_args)
                    result["arg_values"] = sanitized_arg_values

                    previous_layers_results.append({
                        "layer_index": i,
                        "method": current_method,
                        "inferred_state": result
                    })
                    if "field_values" in result and isinstance(result["field_values"], dict):
                        final_payload["field_values"].update(result["field_values"])
                    if i == 0:
                        if "arg_values" in result and isinstance(result["arg_values"], dict):
                            final_payload["arg_values"].update(result["arg_values"])
                        if "description" in result:
                            final_payload["description"] = result["description"]
                    carry_progress_delta = len(result.get("field_values", {})) + len(result.get("arg_values", {}))
                else:
                    carry_progress_delta = 0
                continue

            selected_helpers = []
            selected_class_context = []

            for item in candidate_items:
                feedback = yield {
                    "is_done": False,
                    "decision_type": "context_selection",
                    "layer_index": i,
                    "current_method": current_method,
                    "constraint": layer_goal,
                    "method_code": method_info.get("code", ""),
                    "candidate_item": item,
                    "already_selected": {
                        "helpers": selected_helpers,
                        "class_context": selected_class_context
                    },
                    "progress_delta": 0
                }

                if feedback.get("action") == 1:
                    selected_text = feedback.get("selected_text", item.get("content", ""))
                    source_type = item.get("source_type", "")
                    if source_type.startswith("helper"):
                        selected_helpers.append(selected_text)
                    elif source_type in {"field", "constant", "constructor"}:
                        selected_class_context.append(selected_text)
                    else:
                        selected_class_context.append(selected_text)

            filtered_method_info = {
                "code": method_info.get("code", ""),
                "params": current_params,
                "helpers": "\n\n".join(selected_helpers) if selected_helpers else "(No selected helper methods)",
                "class_context": "\n\n".join(selected_class_context) if selected_class_context else "(No selected class context)"
            }

            prompt = self.build_exploit_state_resolution_prompt(
                i, current_method, next_method, filtered_method_info, layer_goal, sink_requirement, previous_layers_results
            )
            self._dump_resolution_prompt(i, current_method, prompt)
            print(f"[*] Layer {i} | Requesting LLM reasoning...")
            result = self.inferencer.safe_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a JSON-only symbolic engine."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_retries=4,
                retry_base_sleep=2.0,
                fallback_json={"status": "resolved", "arg_values": {}, "field_values": {}}
            )

            if result.get("status") == "resolved":
                sanitized_arg_values = self._sanitize_arg_values(result.get("arg_values", {}), current_params)
                migrated_illegal_args = self._extract_illegal_arg_values_as_fields(result.get("arg_values", {}), current_params)

                if "field_values" not in result or not isinstance(result["field_values"], dict):
                    result["field_values"] = {}
                result["field_values"].update(migrated_illegal_args)
                result["arg_values"] = sanitized_arg_values

                previous_layers_results.append({
                    "layer_index": i,
                    "method": current_method,
                    "inferred_state": result
                })
                if "field_values" in result and isinstance(result["field_values"], dict):
                    final_payload["field_values"].update(result["field_values"])
                if i == 0:
                    if "arg_values" in result and isinstance(result["arg_values"], dict):
                        final_payload["arg_values"].update(result["arg_values"])
                    if "description" in result:
                        final_payload["description"] = result["description"]
                carry_progress_delta = len(result.get("field_values", {})) + len(result.get("arg_values", {}))
            else:
                carry_progress_delta = 0

        yield {
            "is_done": True,
            "final_payload": final_payload,
            "progress_delta": carry_progress_delta
        }

    def resolve_with_context_selector(
        self,
        execution_plan,
        selector,
        slicer,
        context_fallback,
        max_steps: int = 100,
        return_trace: bool = True,
        path_context_pair=None,
    ):
        def is_low_value_text(text: str) -> bool:
            lowered = (text or "").strip().lower()
            markers = [
                "<empty>", "mocked", "fallback", "assumed to", "generic context", "method not found", "timeout",
                "no direct field usage found",
            ]
            return any(m in lowered for m in markers)

        trace = []
        include_count = 0
        exclude_count = 0
        fallback_include_count = 0
        low_value_include_count = 0
        truncated = False
        total_steps = 0
        final_payload = {}

        gen = self.resolve_execution_plan(execution_plan, path_context_pair=path_context_pair)
        event = next(gen)

        while True:
            if event.get("is_done", False):
                final_payload = event.get("final_payload", {})
                break

            total_steps += 1
            if total_steps > max_steps:
                truncated = True
                final_payload = event.get("final_payload", {})
                break

            candidate_item = event["candidate_item"]
            current_method = event["current_method"]
            layer_goal = event["constraint"]
            method_code = event["method_code"]
            already_selected = event.get("already_selected", {})

            action = selector.make_decision(
                layer_goal=layer_goal,
                candidate_item=candidate_item,
                method_code=method_code,
                already_selected=already_selected
            )

            selected_text = ""
            selection_mode = "exclude"

            if action == 1:
                materialized_text = candidate_item.get("content", "") if candidate_item.get("materialized") else ""
                if materialized_text:
                    selected_text = materialized_text
                    selection_mode = "path_context"
                else:
                    sliced_text = slicer.slice_context_item(current_method=current_method, item=candidate_item)
                    if sliced_text and "not found" not in sliced_text.lower() and "timeout" not in sliced_text.lower():
                        selected_text = sliced_text
                        selection_mode = "slice"
                    else:
                        selected_text = context_fallback.generate_fallback_for_context_item(candidate_item)
                        selection_mode = "fallback"
                        fallback_include_count += 1

                if is_low_value_text(selected_text):
                    low_value_include_count += 1

                include_count += 1
            else:
                exclude_count += 1

            if return_trace:
                trace.append({
                    "step": total_steps,
                    "method": current_method,
                    "candidate_signature": candidate_item.get("signature", ""),
                    "candidate_name": candidate_item.get("name", ""),
                    "candidate_source_type": candidate_item.get("source_type", "unknown"),
                    "action": action,
                    "selection_mode": selection_mode,
                    "selected_text_preview": selected_text[:200] if selected_text else "",
                })

            solver_feedback = {
                "action": action,
                "selected_text": selected_text,
                "source_type": candidate_item.get("source_type", "unknown")
            }
            event = gen.send(solver_feedback)

        return {
            "final_payload": final_payload,
            "trace": trace,
            "num_include": include_count,
            "num_exclude": exclude_count,
            "fallback_include_count": fallback_include_count,
            "low_value_include_count": low_value_include_count,
            "num_decisions": include_count + exclude_count,
            "truncated": truncated,
        }
