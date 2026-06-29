# tools/joern_slicer.py
import subprocess
import os
import hashlib
from typing import Dict, Any

class JoernSlicer:
    """
    Extract or slice relevant code for candidate context items.
    """

    def __init__(self, cpg_path: str, joern_bin: str, timeout: int = 60, temp_dir: str = "./tmp"):
        self.cpg_path = cpg_path
        self.joern_bin = joern_bin
        self.timeout = timeout
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        self._method_snippet_cache = {}
        self._field_snippet_cache = {}
    
    def set_cpg_path(self, cpg_path: str):
        self.cpg_path = cpg_path

    def _run_joern_script(self, scala_script: str, key: str) -> str:
        script_name = os.path.join(
            self.temp_dir,
            f"temp_slice_{hashlib.md5(key.encode()).hexdigest()}.sc"
        )

        with open(script_name, "w", encoding="utf-8") as f:
            f.write(scala_script)

        try:
            result = subprocess.run(
                [self.joern_bin, "--script", script_name],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return f"// [Timeout] Joern script timeout after {self.timeout}s."
        except Exception as e:
            return f"// [Error] Joern slicing failed: {str(e)}"
        finally:
            if os.path.exists(script_name):
                os.remove(script_name)

    def get_method_snippet_by_fullname(self, method_fullname: str) -> str:
        cache_key = (self.cpg_path, method_fullname)
        if cache_key in self._method_snippet_cache:
            return self._method_snippet_cache[cache_key]

        safe_fullname = method_fullname.replace('"', '\\"')

        scala_script = f"""
        importCpg("{self.cpg_path}")
        val m = cpg.method.fullNameExact("{safe_fullname}").headOption
        println("===SLICE_START===")
        if (m.isDefined) {{
            val code = m.get.content.getOrElse(m.get.code)
            println(code.split("\\n").take(50).mkString("\\n"))
        }} else {{
            println("// Method not found.")
        }}
        println("===SLICE_END===")
        """

        stdout = self._run_joern_script(scala_script, method_fullname)
        start_idx = stdout.find("===SLICE_START===")
        end_idx = stdout.find("===SLICE_END===")
        if start_idx != -1 and end_idx != -1:
            snippet = stdout[start_idx + len("===SLICE_START==="):end_idx].strip()
            self._method_snippet_cache[cache_key] = snippet
            return snippet

        self._method_snippet_cache[cache_key] = ""
        return ""

    def get_field_related_snippet(self, current_method: str, field_name: str) -> str:
        cache_key = (self.cpg_path, current_method, field_name)
        if cache_key in self._field_snippet_cache:
            return self._field_snippet_cache[cache_key]

        safe_method = current_method.replace('"', '\\"')
        safe_field = field_name.replace('"', '\\"')

        scala_script = f"""
        importCpg("{self.cpg_path}")
        val m = cpg.method.fullNameExact("{safe_method}").headOption
        println("===SLICE_START===")
        if (m.isDefined) {{
            val code = m.get.content.getOrElse(m.get.code)
            val lines = code.split("\\n").filter(l => l.contains("{safe_field}")).take(30)
            if (lines.nonEmpty) println(lines.mkString("\\n"))
            else println("// No direct field usage found in current method.")
        }} else {{
            println("// Current method not found.")
        }}
        println("===SLICE_END===")
        """

        stdout = self._run_joern_script(scala_script, current_method + field_name)
        start_idx = stdout.find("===SLICE_START===")
        end_idx = stdout.find("===SLICE_END===")
        if start_idx != -1 and end_idx != -1:
            snippet = stdout[start_idx + len("===SLICE_START==="):end_idx].strip()
            self._field_snippet_cache[cache_key] = snippet
            return snippet

        self._field_snippet_cache[cache_key] = ""
        return ""

    def slice_context_item(self, current_method: str, item: Dict[str, Any]) -> str:
        source_type = item.get("source_type")
        signature = item.get("signature", "")
        name = item.get("name", "")

        if source_type in {"helper", "helper_fallback", "external_api", "constructor"} and signature:
            # The signature may be plain fallback text, so guard it lightly here.
            if ":" in signature or "<init>" in signature:
                return self.get_method_snippet_by_fullname(signature)

        if source_type in {"field", "constant"} and name:
            return self.get_field_related_snippet(current_method, name)

        if source_type in {"class_context_fallback"}:
            return ""

        return ""
