import subprocess
import os
import json
import re
from pathlib import Path

# This currently uses coarse-grained matching; the matching strategy can still be refined.
SCRIPT_DIR = Path(__file__).resolve().parent

def main():
    # ==== Configuration ====
    CPG_FILE = os.environ.get("VULNSEER_CPG_FILE", str(SCRIPT_DIR / "cpg-file" / "gerenciador-viagens-cpg.bin"))
    SINK = os.environ.get("VULNSEER_SINK", "matches")
    JOERN_BIN = os.environ.get("JOERN_BIN", "joern")
    OUTPUT_FILE = os.environ.get(
        "VULNSEER_CALLCHAINS_OUTPUT",
        str(SCRIPT_DIR / "call-chain-output" / "test-gerenciador-viagens-callchains.json"),
    )
    MAX_DEPTH = 1

    # Generate a Scala script that extracts, normalizes, and prints JSON strings.
    scala_script = f"""
    importCpg("{CPG_FILE}")
    val sink = cpg.method.fullName(".*{SINK}.*")

    def isRelevant(m: Method): Boolean = {{
        !m.fullName.startsWith("java.") && !m.fullName.startsWith("javax.") && !m.fullName.startsWith("sun.")
    }}

    // Extract and normalize call chains.
    val chains = sink.enablePathTracking.repeat(_.caller.filter(isRelevant))(_.maxDepth({MAX_DEPTH}).emit)
      .path
      .map(_.collect {{ case m: Method => m.fullName }}.toList)
      .filter(_.size > 1)   // Filter useless chains.
      .map(_.distinct)      // Deduplicate nodes.
      .dedup.l

    // Reverse bottom-up chains (Sink->Source) into the intuitive top-down order (Source->Sink).
    val reversedChains = chains.map(_.reverse)

    // Manually assemble List[List[String]] into a clean JSON string.
    val jsonStr = reversedChains.map(chain => 
        chain.map(m => "\\"" + m + "\\"").mkString("[", ",", "]")
    ).mkString("[", ",", "]")
    
    // Wrap the JSON with markers so Python can extract it precisely.
    println("===JSON_START===")
    println(jsonStr)
    println("===JSON_END===")
    """

    script_name = "query_chain.sc"
    with open(script_name, "w", encoding="utf-8") as f:
        f.write(scala_script)

    print(f"[+] Querying Joern and generating JSON (target: {CPG_FILE}) ...")
    
    try:
        # Capture subprocess output.
        result = subprocess.run(
            [JOERN_BIN, "--script", script_name],
            capture_output=True,
            text=True
        )
        
        stdout_str = result.stdout
        
        # Extract the JSON payload precisely.
        match = re.search(r'===JSON_START===\s*(.*?)\s*===JSON_END===', stdout_str, re.DOTALL)
        
        if match:
            json_data_str = match.group(1)
            paths = json.loads(json_data_str)
            
            # Save to a local file.
            Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(paths, f, indent=4, ensure_ascii=False)
            
            print(f"[+] Analysis completed. Extracted {len(paths)} call chains.")
            print(f"[+] Saved successfully to {OUTPUT_FILE}")
            
        else:
            print("[-] Failed to extract JSON data from Joern output.")
            print("[*] Raw Joern output:")
            print(stdout_str)
            if result.stderr:
                print("[*] Error output:")
                print(result.stderr)
                
    except Exception as e:
        print(f"[-] Execution failed: {e}")
    finally:
        # Remove the temporary Scala script.
        if os.path.exists(script_name):
            os.remove(script_name)

if __name__ == "__main__":
    main()
