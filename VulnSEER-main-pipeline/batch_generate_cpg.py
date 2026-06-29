import subprocess
import os
from pathlib import Path

# =========================
# Configuration
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
CLIENT_APPS_ROOT = Path(os.environ.get("VULNSEER_CLIENT_APPS_ROOT", SCRIPT_DIR / "client-apps")).expanduser()
CPG_OUTPUT_DIR = Path(os.environ.get("VULNSEER_CPG_OUTPUT_DIR", SCRIPT_DIR / "cpg-file")).expanduser()

# Use an absolute path here if joern-parse is not on PATH.
JOERN_PARSE_BIN = os.environ.get("JOERN_PARSE_BIN", "joern-parse")

client_projects_path = [
    "adu-test",
    "adyen-api",
    "archivefs",
    "axon-server-se/axonserver",
    "bean-query",
    "CarStoreApi/account/account-web",
    "commons-validator",
    "db/engine",
    "elasticsearch-maven-plugin",
    "flow",
    "geek-framework",
    "gerenciador-viagens",
    "huntfiles",
    "idworker",
    "jerry-core",
    "jfinal",
    "JsonConfiguration",
    "kafka-keyvalue",
    "karate/karate-core",
    "knetbuilder/ondex-base/core/marshal",
    "mirage/mirage-core",
    "Mixmicro-Components/llc-kits",
    "neo",
    "ninja/ninja-core",
    "OmegaTester",
    "Online_Train_Ticket_Reservation_System/Code",
    "PatentPublicData/Common",
    "pdf-converter",
    "pdf-util",
    "PLMCodeTemplate/source",
    "QLExpress",
    "reproducible-build-maven-plugin",
    "rike/arago-rike-commons",
    "roubsite/RoubSiteUtils",
    "rpki-commons",
    "RuoYi-Vue-Multi-Tenant/multi-tenant-server",
    "serritor",
    "son-editor/son-validate-web",
    "tcpser4j",
    "twirl",
    "ucloud-java-sdk",
    "UltraPlaytime",
    "WxJava/weixin-java-mp",
    "wakatime-sync",
    "webbit",
    "weblaf/modules/core",
    "base-starter",
    "wechat-ssm",
    "ZingClient"
]

# =========================
# Helper functions
# =========================
def get_cpg_name(project_rel_path: str) -> str:
    """
    Use the deepest directory name as the CPG file name.
    Examples:
      mirage/mirage-core -> mirage-core-cpg.bin
      PLMCodeTemplate/source -> source-cpg.bin
    """
    deepest_dir_name = Path(project_rel_path).name
    return f"{deepest_dir_name}-cpg.bin"


def run_joern_parse(project_abs_path: Path, cpg_output_path: Path) -> bool:
    cmd = [
        JOERN_PARSE_BIN,
        str(project_abs_path),
        "--language", "javasrc",
        "--output", str(cpg_output_path),
        "--frontend-args", "--enable-file-content"
    ]

    print(f"\n[+] Start generating CPG")
    print(f"    Project path: {project_abs_path}")
    print(f"    Output path: {cpg_output_path}")
    print(f"    Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[-] Generation failed: {project_abs_path}")
        print(f"    Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"[-] Command not found: {JOERN_PARSE_BIN}")
        print("    Make sure joern-parse is on PATH, or set JOERN_PARSE_BIN to an absolute path.")
        return False


def main():
    CPG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing CPG files dynamically to avoid maintaining a generated-file list by hand.
    existing_cpg_files = {p.name for p in CPG_OUTPUT_DIR.glob("*-cpg.bin")}

    total = len(client_projects_path)
    skipped = 0
    success = 0
    failed = 0
    not_found = 0

    print(f"[*] Output directory: {CPG_OUTPUT_DIR}")
    print(f"[*] Existing CPG count: {len(existing_cpg_files)}")
    print(f"[*] Projects to check: {total}")

    for idx, rel_path in enumerate(client_projects_path, 1):
        project_abs_path = CLIENT_APPS_ROOT / rel_path
        cpg_name = get_cpg_name(rel_path)
        cpg_output_path = CPG_OUTPUT_DIR / cpg_name

        print(f"\n{'=' * 80}")
        print(f"[*] [{idx}/{total}] Processing project: {rel_path}")
        print(f"[*] Target CPG: {cpg_name}")

        if not project_abs_path.exists():
            print(f"[-] Project path does not exist, skipping: {project_abs_path}")
            not_found += 1
            continue

        if not project_abs_path.is_dir():
            print(f"[-] Target is not a directory, skipping: {project_abs_path}")
            not_found += 1
            continue

        if cpg_name in existing_cpg_files or cpg_output_path.exists():
            print(f"[=] Already exists, skipping: {cpg_output_path}")
            skipped += 1
            continue

        ok = run_joern_parse(project_abs_path, cpg_output_path)
        if ok:
            print(f"[+] Generation succeeded: {cpg_output_path}")
            success += 1
        else:
            print(f"[x] Generation failed: {project_abs_path}")
            failed += 1

    print(f"\n{'=' * 80}")
    print("[*] All projects processed.")
    print(f"    Total projects     : {total}")
    print(f"    Generated          : {success}")
    print(f"    Skipped existing   : {skipped}")
    print(f"    Invalid paths      : {not_found}")
    print(f"    Failed generations : {failed}")


if __name__ == "__main__":
    main()
