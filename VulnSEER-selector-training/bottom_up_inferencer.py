
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
    # def __init__(self, cpg_path, joern_bin, model="deepseek-chat", debug_prompt_dir="./output/debug_prompts"):
    #     self.cpg_path = cpg_path
    #     self.joern_bin = joern_bin
    #     self.model = model
    #     self.debug_prompt_dir = debug_prompt_dir
    #     os.makedirs(self.debug_prompt_dir, exist_ok=True)

    #     api_key = os.environ.get("DEEPSEEK_API_KEY")
    #     if not api_key:
    #         print("[-] 警告: 未找到 DEEPSEEK_API_KEY 环境变量。")

    #     self.client = OpenAI(
    #         api_key=api_key,
    #         base_url="https://api.deepseek.com"
    #     )
    #     self._method_context_cache = {}
    def __init__(self, cpg_path, joern_bin, model="gpt-5.2", debug_prompt_dir="./output/debug_prompts"):
        self.cpg_path = cpg_path
        self.joern_bin = joern_bin
        self.model = model
        self.debug_prompt_dir = debug_prompt_dir
        os.makedirs(self.debug_prompt_dir, exist_ok=True)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[-] 警告: 未找到 API_KEY 环境变量。")

        if OpenAI is None:
            raise ImportError("缺少 openai Python 包；请先在运行环境中安装: pip install openai")

        self.client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")
        self._method_context_cache = {}
        
    def set_cpg_path(self, cpg_path: str):
        self.cpg_path = cpg_path

    def parse_legacy_helper_text(self, raw_text: str):
        """
        兼容旧版 raw helper 文本解析
        输出统一的 source_type='helper'
        """
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
                "source_type": "helper",   # 统一成 helper
                "name": helper_name,
                "signature": signature,
                "content": content
            })

        return items

    def parse_legacy_class_context_text(self, raw_text: str):
        """
        兼容旧版 raw class context 文本解析
        统一输出 source_type in {field, constant, constructor}
        """
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

            # 先处理 constructor block
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

            # 再处理字段行
            for line in lines:
                if line.startswith("Field:"):
                    content = line.replace("Field:", "").strip()
                    source_type = "constant" if "static final" in content else "field"

                    # 尽量提取变量名
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
        """
        规范化候选上下文项：

        规则：
        1. 如果是 external method，直接跳过本地上下文归一化，返回空候选项
        2. 对项目内方法：
        - 优先使用结构化 helpers / class_context
        - 仅在结构化结果为空时才 fallback 到 raw 文本解析
        3. 做去空、去重、去噪、限长
        """

        # ---------------------------------------------------------
        # 0) external method：不做本地上下文候选项选择
        # ---------------------------------------------------------
        if method_info.get("is_external", False):
            print("[DEBUG:normalize] external method -> skip local helper/class-context normalization")
            return {
                "helpers": [],
                "class_context": []
            }

        helpers = method_info.get("helpers", []) or []
        class_context = method_info.get("class_context", []) or []

        print(f"[DEBUG:normalize] before cleanup helpers={len(helpers)}, class_context={len(class_context)}")

        # ---------------------------------------------------------
        # 1) 单项有效性判断
        # ---------------------------------------------------------
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

            # 明显空壳
            if lowered_content in {"<empty>", "unknown", "none", "null"}:
                return False

            # 明显 Joern operator / 噪声
            noisy_keywords = [
                "<operator>",
                "logger",
                "log.",
                "println",
                "printstacktrace",
                "tostring",
                "hashcode",
                "equals(",
            ]
            if any(k in lowered_sig for k in noisy_keywords):
                return False
            if any(k in lowered_content for k in ["logger", "log.", "println", "printstacktrace"]):
                return False

            # getter / setter 通常不是关键上下文
            if lowered_name.startswith("get") or lowered_name.startswith("set"):
                return False

            # fallback 里常见的空 constructor 外壳
            if lowered_name == "<init>" and lowered_content == "<empty>":
                return False
            
            # 过滤纯默认空构造函数：只做 this 绑定 + 调父类构造 + return
            if source_type == "constructor":
                normalized = " ".join(lowered_content.split())
                if (
                    "specialinvoke this.<java.lang.object: void <init>()>()" in normalized
                    and "return;" in normalized
                    and normalized.count("specialinvoke") == 1
                ):
                    # 没有字段赋值、没有业务逻辑、没有配置初始化
                    constructor_signal_keywords = [
                        " = ",
                        "put(",
                        "add(",
                        "config",
                        "handler",
                        "factory",
                        "session",
                        "request",
                        "response",
                        "token",
                        "password",
                        "encoder",
                        "decoder"
                    ]
                    if not any(k in lowered_content for k in constructor_signal_keywords):
                        return False

            # 对字段做一点保守过滤：纯常见无关字段可去掉
            if source_type in {"field", "constant"}:
                low_value_field_keywords = ["serialversionuid", "logger", "log"]
                if lowered_name in low_value_field_keywords:
                    return False

            return True

        # ---------------------------------------------------------
        # 2) 去重
        # ---------------------------------------------------------
        def dedup_items(items):
            seen = set()
            deduped = []
            for item in items:
                key = (
                    item.get("source_type", ""),
                    item.get("signature", ""),
                    item.get("content", "")
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            return deduped

        # ---------------------------------------------------------
        # 3) 规范 source_type（兼容 helper_fallback / ctor_fallback 等）
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 4) 先清洗结构化结果
        # ---------------------------------------------------------
        helpers = [x for x in helpers if is_meaningful_item(x)]
        class_context = [x for x in class_context if is_meaningful_item(x)]

        helpers = dedup_items(helpers)
        class_context = dedup_items(class_context)

        print(f"[DEBUG:normalize] after structured cleanup helpers={len(helpers)}, class_context={len(class_context)}")

        # ---------------------------------------------------------
        # 5) 仅当项目内方法的结构化结果为空时才 fallback
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 6) 再做一次按类型过滤，确保 helper / class_context 分类干净
        # ---------------------------------------------------------
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

        # 如果 fallback 解析后 helper 被错误归类，做一个轻微补救
        for item in helpers:
            st = item.get("source_type", "")
            if st not in {"helper", "field", "constant", "constructor"}:
                # 根据 name/signature 做一个保守判断
                sig = (item.get("signature") or "").lower()
                name = (item.get("name") or "").lower()
                if "<init>" in sig or name == "<init>":
                    item["source_type"] = "constructor"
                    final_class_context.append(item)
                else:
                    item["source_type"] = "helper"
                    final_helpers.append(item)

        # ---------------------------------------------------------
        # 7) 限长，避免 prompt 爆炸
        # ---------------------------------------------------------
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

        return {
            "helpers": final_helpers,
            "class_context": final_class_context
        }

    def _get_smart_project_prefix(self, chain):
        prefixes = []
        blacklist = {'java', 'javax', 'org.springframework', 'com.sun', 'org.apache', 'jakarta'}

        for sig in chain:
            package_path = sig.split(':')[0]
            parts = package_path.split('.')
            if len(parts) > 2:
                prefix_2 = ".".join(parts[:2])
                prefix_3 = ".".join(parts[:3])

                if not any(prefix_2.startswith(b) for b in blacklist):
                    prefixes.append(prefix_2)
                    prefixes.append(prefix_3)

        if not prefixes:
            return ""

        from collections import Counter
        most_common = Counter(prefixes).most_common(1)
        return most_common[0][0] if most_common else ""

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
        env["JAVA_HOME"] = ""
        env["PATH"] = f"" + env.get("PATH", "")

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
                print(f"[!] 警告: CPG 中未找到 {method_fullname}")
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

            # -------------------------
            # 1) 解析 helpers
            # -------------------------
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

                # 过滤一部分明显无关/噪声 helper
                lowered_sig = signature.lower()
                if any(k in lowered_sig for k in ["println", "logger", "log.", "tostring"]):
                    continue

                # 名字提取
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

            # -------------------------
            # 2) 解析 class_context
            # -------------------------
            class_items = []
            class_idx = 0

            class_blocks = [b.strip() for b in raw_class.split("---") if b.strip()]
            for block in class_blocks:
                lines = [x.rstrip() for x in block.splitlines() if x.strip()]
                if not lines:
                    continue

                # 构造函数块
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

                # 字段行
                for line in lines:
                    if line.startswith("Field:"):
                        field_code = line.replace("Field:", "").strip()
                        source_type = "constant" if "static final" in field_code else "field"

                        # 简单抽变量名
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
            print(f"[-] Joern 查询超时: {method_fullname}")
            return method_info
        except Exception as e:
            print(f"[-] 提取过程出错: {e}")
            self._method_context_cache[cache_key] = method_info
            return json.loads(json.dumps(method_info))

    def generate_sink_requirement_from_exploit_sketch(self, exploit_sketch_dir, sink_method):
        print(f"\n[*] 正在从目录 {exploit_sketch_dir} 搜索并分析 sink-level exploit sketch...")
        if not os.path.exists(exploit_sketch_dir):
            raise FileNotFoundError(f"目录不存在: {exploit_sketch_dir}")

        java_files = [f for f in os.listdir(exploit_sketch_dir) if f.endswith(".java")]
        if not java_files:
            raise FileNotFoundError(f"在 {exploit_sketch_dir} 中未找到任何 .java exploit sketch 文件！")

        target_file = os.path.join(exploit_sketch_dir, java_files[0])
        print(f"[*] 选定 exploit sketch 文件: {target_file}")

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

The provided exploit sketch only represents ONE specific instance of the exploit. You must GENERALIZE the payload. Abstract the core malicious characteristics required to trigger the vulnerability class (e.g., DoS, RCE, Path Traversal) rather than copying the exact hardcoded string.

Formulate these characteristics into constraints for the arguments of the Sink Method.

=== OUTPUT FORMAT ===
You MUST return a SINGLE valid JSON object representing the constraints for the sink method's parameters.
The JSON schema MUST follow this exact format (do NOT add extra keys outside of arg0, arg1, etc.):
{{
"arg0": {{
"type": "[Fully qualified Java type, e.g., java.lang.String]",
"placeholder": "<SINK_ARG0>",
"description": "[Your generalized description of the malicious payload characteristics based on the testcase]"
}}
}}
"""
        print(f"[*] 请求 LLM 泛化 sink-level requirement...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a JSON-only vulnerability analyzer. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result_text = response.choices[0].message.content
        try:
            sink_requirement = json.loads(result_text)
            print(f"[+] Sink-level requirement 泛化成功:\n{json.dumps(sink_requirement, indent=2, ensure_ascii=False)}")
            return sink_requirement
        except json.JSONDecodeError:
            print("[-] LLM 解析 sink-level requirement 失败，返回的不是合法的 JSON。")
            raise RuntimeError("Sink requirement analysis failed.")

    def build_prompt(self, caller, callee, method_info, callee_requirement, is_lowest_layer):
        """
        动态注入所有提取到的上下文信息，组装 Prompt
        """
        params = method_info.get("params", [])
        param_str = f"The caller method fi has {len(params)} parameter(s).\n"
        for i, p in enumerate(params):
            param_str += f"  - arg{i}: {p['name']} ({p['type']})\n"

        req_str = json.dumps(callee_requirement, indent=2)

        sink_requirement_section = ""
        if is_lowest_layer:
            sink_requirement_section = f"""=== Sink-Level Requirement ===
This is the lowest layer (vulnerable sink).
An exploit sketch has been extracted. The vulnerable sink must receive the following arguments:
{req_str}
When you want fi-1 to receive exactly the same runtime object as in the exploit sketch,
you MUST use the placeholder string "<SINK_ARGi>" in candidate_inputs.args...
"""

        # 获取提取的上下文，处理空值
        source_code = method_info.get("code") or "(Code not available)"
        helpers = self._render_context_items(method_info.get("helpers", []))
        class_context = self._render_context_items(method_info.get("class_context", []))

        prompt = f"""You are a symbolic execution engine for Java.
You perform Hoare-logic style reasoning *for a single step in a call chain*.

=== Current Layer ===
We are analyzing one level of a call chain, specifically:
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
1. [Context Analysis]: Analyze the 'Helper Methods' and 'Class Context' above. 
   - Does the caller use any static constants or private helper methods to process the input?
   - Does the constructor initialize any fields (e.g., this.config) that affect the logic?
2. [Precondition Inference]: Based on the analysis,you must produce *caller-level preconditions* such that:
  (1). When fi executes under these preconditions,
  (2). It will invoke fi-1 with arguments satisfying the above constraints.
3. [Critical Rule for Object Construction]: You cannot define method behaviors(like getHeaders) via JSON keys. If a field is an Interface(e.g., HttpRequest), you MUST:
  (1). Identify a concrete implementation class(e.g., DefaultHttpRequest).
  (2). Construct that concrete object structure in JSON.
  (3). Use the concrete class internal fields (e.g., headers map) to store the data.

=== OUTPUT FORMAT (VERY IMPORTANT)=== 
You MUST return a SINGLE valid JSON object, with NO extra text before or after it.
The JSON schema is:
{{
  "caller_precondition": ["...", "..."],
  "candidate_inputs": [
    {{"args": ["arg0_value", "arg1_value"]}}
  ]
}}

=== Constraints === 
- The output MUST be valid JSON.
- Do NOT use any code expressions such as new Object(), new String("x"), constructors, or method calls.
- Each argument in "args" MUST be a plain JSON literal: number, string, boolean, null, array, or object.
- If an argument cannot be represented literally (e.g., HttpRequestWrapper), use a STRING description instead, e.g., "<HttpRequestWrapper>".
- Do NOT include comments, trailing commas, or any text outside the JSON.
- The "args" array in each element of "candidate_inputs" MUST have exactly {max(1, len(params))} element(s), matching the caller's parameters in order.
- If the caller has 0 parameters, "args" MUST be an empty array [].
- Do NOT invent extra arguments that are not in the caller's parameter list, and do NOT omit required parameters.
- If some constraints involve internal fields or configuration of fi (not method parameters), express them ONLY in "caller_precondition", NOT as extra elements in "args".
- If you need to reuse the exact sink argument object, use "<SINK_ARG0>", "<SINK_ARG1>", etc.,
  instead of copying the toString() output like "java.lang.Object@4b18f2b1".
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
        """
        统一的带重试 LLM 调用。
        - 网络抖动 / 连接重置 / 限流时自动重试
        - 多次失败后返回 fallback_json，而不是让整个训练崩掉
        """
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
                content = response.choices[0].message.content
                return json.loads(content)

            except json.JSONDecodeError as e:
                print(f"[-] LLM 返回了非法 JSON (attempt={attempt}/{max_retries}): {e}")
                last_err = e

            except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
                print(f"[-] LLM 调用失败 (attempt={attempt}/{max_retries}): {type(e).__name__}: {e}")
                last_err = e

            except Exception as e:
                print(f"[-] LLM 未知异常 (attempt={attempt}/{max_retries}): {type(e).__name__}: {e}")
                last_err = e

            if attempt < max_retries:
                sleep_sec = retry_base_sleep * attempt
                print(f"[*] {sleep_sec:.1f}s 后重试...")
                time.sleep(sleep_sec)

        print(f"[!] LLM 连续失败，返回 fallback 结果: {fallback_json}")
        return fallback_json


    # def call_llm(self, prompt):
    #     """
    #     调用 LLM API，强制返回 JSON
    #     """
    #     print(f"[*] 请求 LLM ({self.model}) 推导当前层约束...")
    #     response = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=[
    #             {"role": "system", "content": "You are a JSON-only symbolic execution engine. Output strictly valid JSON."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         response_format={"type": "json_object"},
    #         temperature=0.2 
    #     )
        
    #     result_text = response.choices[0].message.content
    #     try:
    #         return json.loads(result_text)
    #     except json.JSONDecodeError:
    #         print("[-] LLM 返回的不是合法的 JSON。")
    #         return {"error": "Invalid JSON", "raw": result_text}
    def call_llm(self, prompt):
        """
        调用 LLM API，强制返回 JSON；带自动重试与失败兜底
        """
        print(f"[*] 请求 LLM ({self.model}) 推导当前层约束...")
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

    def infer_layerwise_exploit_goals(self, chain, sink_requirement):
        """
        执行 layer-wise exploit-goal inference
        """
        print(f"\n[+] 开始 layer-wise exploit-goal inference (Chain length: {len(chain)})")
        exploit_goal_map = {}
        
        sink_method = chain[-1]
        current_requirement = sink_requirement
        exploit_goal_map[sink_method] = current_requirement

        # 创建用于保存 debug prompt 的目录
        debug_dir = "./output/debug_prompts"
        os.makedirs(debug_dir, exist_ok=True)

        # 逆向遍历：从倒数第二个方法开始
        for i in range(len(chain) - 2, -1, -1):
            step_num = len(chain) - 1 - i 
            caller = chain[i]
            callee = chain[i + 1]
            is_lowest_layer = (i == len(chain) - 2)
            
            print(f"\n--- 层级分析 (Step {step_num}): {caller} -> {callee} ---")
            
            # 1. 提取上下文
            method_info = self.get_method_context_from_cpg(caller)
            if not method_info.get("code"):
                print(f"[!] 警告: 未能在 CPG 中找到 {caller} 的源码。")
            
            # 2. 组装 Prompt
            prompt = self.build_prompt(caller, callee, method_info, current_requirement, is_lowest_layer)
            
            # 3. 记录 Debug Prompt 文件
            # 提取干净的方法名用于做文件名
            safe_caller_name = re.sub(r'[^a-zA-Z0-9_\.]', '_', caller.split(':')[0].split('.')[-2] + '.' + caller.split(':')[0].split('.')[-1])
            debug_file_path = os.path.join(debug_dir, f"step_{step_num}_{safe_caller_name}_prompt.txt")
            
            with open(debug_file_path, "w", encoding="utf-8") as df:
                df.write(prompt)
            print(f"[*] Prompt 已保存至: {debug_file_path}")
            
            # 4. 请求大模型求解
            llm_result = self.call_llm(prompt)
            print(f"[+] 推导结果:\n{json.dumps(llm_result, indent=2, ensure_ascii=False)}")
            
            # 5. 更新exploit goal map
            exploit_goal_map[caller] = llm_result
            current_requirement = llm_result

        return exploit_goal_map


# ==========================================
# Framework stage: execution-plan construction
# ==========================================

class ExecutionPlanBuilder:
    """
    将 layer-wise exploit goals 组装成 entry-to-sink execution plan.
    """
    @staticmethod
    def build_execution_plan(chain, exploit_goal_map):
        print("\n[+] 开始构造 entry-to-sink execution plan...")
        execution_plan = []
        
        # 按照从入口点(Entry)到漏洞点(Sink)的正向顺序遍历
        for i, method in enumerate(chain):
            layer_info = {
                "step_index": i,
                "method_signature": method,
            }
            
            # 1. 确定当前层的角色标签
            if i == 0:
                layer_info["layer_type"] = "ENTRY"
            elif i == len(chain) - 1:
                layer_info["layer_type"] = "SINK"
            else:
                layer_info["layer_type"] = "INTERMEDIATE"
                
            # 2. 确定下一跳目标 (Callee)
            if i < len(chain) - 1:
                layer_info["calls_next"] = chain[i + 1]
            else:
                layer_info["calls_next"] = None

            # 3. 挂载阶段一推导出的约束条件
            goal = exploit_goal_map.get(method, {})
            
            if layer_info["layer_type"] == "SINK":
                # Sink 层直接保存泛化后的 sink requirement 要求
                layer_info["sink_requirements"] = goal
            else:
                # Caller 层保存前置条件和候选载荷
                layer_info["preconditions"] = goal.get("caller_precondition", [])
                layer_info["candidate_inputs"] = goal.get("candidate_inputs", [])
                
            execution_plan.append(layer_info)
            
        print(f"[*] 成功组装 execution plan，共包含 {len(execution_plan)} 个 plan unit。")
        return execution_plan

# ==========================================
# Framework stage: LLM-guided exploit-state resolution
# ==========================================

class ExploitStateResolver:
    """
    新版：
    在每层真正求解前，先对 helpers + class_context 每个候选上下文项逐项 yield 给 RL 环境
    """
    def __init__(self, inferencer):
        self.inferencer = inferencer
        self.client = inferencer.client
        self.model = inferencer.model
    
    def set_cpg_path(self, cpg_path: str):
        self.inferencer.set_cpg_path(cpg_path)

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
        param_str = "\n".join([f"  - arg{i}: {p['name']} ({p['type']})" for i, p in enumerate(params)])
        source_code = method_info.get("code") or "(Code not available)"
        helpers = method_info.get("helpers") or "(No selected helper methods)"
        class_context = method_info.get("class_context") or "(No selected class context)"

        goal_str = json.dumps(layer_goal, indent=2, ensure_ascii=False)
        sink_requirement_str = json.dumps(sink_requirement, indent=2, ensure_ascii=False)
        prev_states_str = json.dumps(previous_layers_results, indent=2, ensure_ascii=False) if previous_layers_results else "(None)"

        schema_description = ""
        if layer_idx == 0:
            schema_description = '  "description": "<A detailed natural language explanation of how this entry payload triggers the vulnerability based on the sink requirement>",\n'

        prompt = f"""You are a symbolic execution engine for Java.
We are analyzing LAYER INDEX {layer_idx} of the call chain.
Method signature: {current_method}

=== Parameters ===
{param_str}

=== Method Source Code ===
{source_code}

=== Helper Methods Source (Selected by RL Agent) ===
{helpers}

=== Class Context (Selected by RL Agent) ===
{class_context}

=== Sink-Level Requirement ===
The placeholders like <SINK_ARG0> originate from these root constraints at the vulnerable sink:
{sink_requirement_str}

=== Execution State from Previous (Higher) Layers ===
These are the parameter values and field states already determined by the layers above the current one. You must maintain consistency with these states:
{prev_states_str}

=== Layer Exploit Goal ===
{goal_str}

=== Critical Reasoning & Action Guidelines ===
You must analyze if you can fully determine the required parameters and hidden fields to reach the Goal.
1. **Request Help if Stuck**: If you encounter an implicit variable, field (e.g., `this.config`), or parameter whose initialization is unknown, AND it blocks your reasoning, you MUST request help from the RL Agent.
2. **Instantiate Concrete Values (NO PLACEHOLDERS)**: If you have enough information, you MUST generate REALISTIC, CONCRETE malicious payloads (e.g., actual SQL injection strings, path traversal paths) based on the sink-level requirement. DO NOT output placeholders like "<SINK_ARG0>".
3. **Consistency**: Ensure your outputs do not logically conflict with the 'Execution State from Previous Layers'.

=== Output JSON Schema ===
You must infer concrete values.
Output one JSON object:
{{
  "status": "resolved",
{schema_description}  "arg_values": {{ "arg0": "<CONCRETE value, NO placeholders>", "arg1": "<CONCRETE value>" }},
  "field_values": {{ "fieldName1": "<concrete value>" }}
}}
"""
        return prompt
    
    def resolve_execution_plan(self, execution_plan):
        """
        核心生成器：
        - 项目内层：提取 helpers/class_context，逐项 yield 给 RL 决策
        - 外部库层：不做上下文项选择，直接进入 solver
        - 项目内但无候选项：也直接进入 solver
        """
        if not execution_plan or execution_plan[-1].get("layer_type") != "SINK":
            raise ValueError("execution_plan 非法：最后一层必须是 SINK")

        sink_requirement = execution_plan[-1].get("sink_requirements", {})
        if not sink_requirement:
            raise ValueError("execution_plan 缺少 sink_requirements")

        final_payload = {"arg_values": {}, "field_values": {}}
        previous_layers_results = []
        carry_progress_delta = 0

        for i, layer in enumerate(execution_plan):
            current_method = layer["method_signature"]
            next_method = layer.get("calls_next") or "Sink"

            # 先定义 layer_goal，后续所有分支都能用
            layer_goal = {
                "preconditions": layer.get("preconditions", []),
                "candidate_inputs": layer.get("candidate_inputs", [])
            }

            method_info = self.inferencer.get_method_context_from_cpg(current_method)

            # =========================================================
            # 分支 A：外部库层 -> 不做上下文项选择，直接进入 solver
            # =========================================================
            if method_info.get("is_external", False):
                print(f"[DEBUG] Layer {i} | external method detected: {current_method}")

                filtered_method_info = {
                    "code": "(External library method implementation not available in current project CPG)",
                    "params": method_info.get("params", []),
                    "helpers": "(No helper methods available for external library method)",
                    "class_context": f"(No class context available: {method_info.get('no_context_reason', 'external method')})"
                }

                prompt = self.build_exploit_state_resolution_prompt(
                    i,
                    current_method,
                    next_method,
                    filtered_method_info,
                    layer_goal,
                    sink_requirement,
                    previous_layers_results
                )

                print(f"[*] Layer {i} | 请求 LLM 推理...")
                
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
                    print(f"[+] Layer {i} | external layer resolved.")
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

            # =========================================================
            # 分支 B：项目内层 -> 结构化候选项清洗
            # =========================================================
            normalized = self.inferencer.normalize_candidate_items(method_info)

            structured_helper_count = len(method_info.get("helpers", []))
            structured_class_count = len(method_info.get("class_context", []))

            print(
                f"[DEBUG] Layer {i} | raw structured helpers={structured_helper_count}, "
                f"raw structured class_context={structured_class_count}"
            )
            print(f"[DEBUG] Layer {i} | method={current_method}")

            candidate_items = normalized["helpers"] + normalized["class_context"]
            print(
                f"[DEBUG] helpers={len(normalized['helpers'])}, "
                f"class_context={len(normalized['class_context'])}, "
                f"total_candidates={len(candidate_items)}"
            )

            # =========================================================
            # 分支 C：项目内层但没有候选项 -> 不做 RL，直接进入 solver
            # =========================================================
            if not candidate_items:
                print(f"[DEBUG] Layer {i} | project-local method but no candidate context items.")

                filtered_method_info = {
                    "code": method_info.get("code", ""),
                    "params": method_info.get("params", []),
                    "helpers": "(No helper methods extracted)",
                    "class_context": "(No class context extracted)"
                }

                prompt = self.build_exploit_state_resolution_prompt(
                    i,
                    current_method,
                    next_method,
                    filtered_method_info,
                    layer_goal,
                    sink_requirement,
                    previous_layers_results
                )

                print(f"[*] Layer {i} | 请求 LLM 推理...")
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
                    print(f"[+] Layer {i} | local-no-context layer resolved.")
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

            # =========================================================
            # 分支 D：项目内层且有候选项 -> 逐项 RL 决策
            # =========================================================
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
                "params": method_info.get("params", []),
                "helpers": "\n\n".join(selected_helpers) if selected_helpers else "(No selected helper methods)",
                "class_context": "\n\n".join(selected_class_context) if selected_class_context else "(No selected class context)"
            }

            prompt = self.build_exploit_state_resolution_prompt(
                i,
                current_method,
                next_method,
                filtered_method_info,
                layer_goal,
                sink_requirement,
                previous_layers_results
            )

            print(f"[*] Layer {i} | 请求 LLM 推理...")
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
                print(f"[+] Layer {i} | 成功推导当前层状态。")
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


