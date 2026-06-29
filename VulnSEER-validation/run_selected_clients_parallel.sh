#!/usr/bin/env bash

# ========= Basic configuration =========

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# VulnSEER-validation JAR path
JAR_PATH="$PROJECT_ROOT/vulnseer-engine/target/vulnseer-engine-1.0.jar"

# client-apps root directory
CLIENT_BASE="$PROJECT_ROOT/client-apps"

# Output root directory. Each app gets its own subdirectory.
OUTPUT_BASE="$PROJECT_ROOT/output"

# Selected client apps to validate
APPS=(
  "axon-server-se/axonserver"
  "db/engine"
  "flow"
)

# Maximum number of parallel jobs
MAX_JOBS=3

# Log directory
LOG_DIR="$PROJECT_ROOT/vulnseer_logs"

# ========= Script entry point =========

set -euo pipefail

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_BASE"

echo "========== VulnSEER-validation Parallel Runner =========="
echo "[INFO] JAR_PATH     = $JAR_PATH"
echo "[INFO] CLIENT_BASE  = $CLIENT_BASE"
echo "[INFO] OUTPUT_BASE  = $OUTPUT_BASE"
echo "[INFO] MAX_JOBS     = $MAX_JOBS"
echo "[INFO] LOG_DIR      = $LOG_DIR"
echo "============================================"
echo

if [ ! -f "$JAR_PATH" ]; then
  echo "[ERROR] JAR file not found: $JAR_PATH"
  exit 1
fi

for app_name in "${APPS[@]}"; do
  app_dir="$CLIENT_BASE/$app_name"
  log_file="$LOG_DIR/${app_name//\//__}.log"

  # Use one output directory per app: OUTPUT_BASE/app_name
  app_output="$OUTPUT_BASE/$app_name"
  mkdir -p "$app_output"

  if [ ! -d "$app_dir" ]; then
    echo "[WARN] Skip missing app directory: $app_dir"
    continue
  fi

  echo "[INFO] Start job: $app_name"
  echo "[INFO]   -p      $app_dir"
  echo "[INFO]   -output $app_output"
  echo "[INFO]   log:    $log_file"

  java -noverify -jar "$JAR_PATH" \
    -p "$app_dir" \
    -output "$app_output" \
    >"$log_file" 2>&1 &

  # Limit the number of concurrent jobs.
  while [ "$(jobs -r | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do
    sleep 2
  done
done

echo
echo "[INFO] All jobs have been started. Waiting for completion..."
wait
echo "[INFO] All selected client apps completed."
echo "[INFO] Per-app output directories are under: $OUTPUT_BASE/<app-name>/"
