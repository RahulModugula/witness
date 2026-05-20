#!/usr/bin/env bash
# End-to-end smoke test against the running service.
# Usage: ./scripts/demo.sh [host:port]
set -euo pipefail

HOST="${1:-localhost:8000}"
SAMPLE="${SAMPLE:-data/uploads/sample.mp4}"

if [[ ! -f "$SAMPLE" ]]; then
  echo "Sample video not found at $SAMPLE" >&2
  exit 1
fi

JQ="$(command -v jq || true)"
pretty() { if [[ -n "$JQ" ]]; then jq .; else cat; fi; }

echo "→ /health"
curl -fsS "http://$HOST/health" | pretty
echo

echo "→ POST /tasks (uploading $SAMPLE)"
CREATE_JSON="$(curl -fsS -X POST "http://$HOST/tasks" -F "file=@$SAMPLE")"
echo "$CREATE_JSON" | pretty
TASK_ID="$(echo "$CREATE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin)["task_id"])')"
echo "task_id=$TASK_ID"
echo

echo "→ polling /tasks/$TASK_ID"
for i in $(seq 1 120); do
  STATUS_JSON="$(curl -fsS "http://$HOST/tasks/$TASK_ID")"
  STATUS="$(echo "$STATUS_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')"
  printf "  [%02ds] %s\n" "$i" "$STATUS"
  case "$STATUS" in
    DONE|FAILED) break ;;
  esac
  sleep 1
done

if [[ "$STATUS" != "DONE" ]]; then
  echo "task did not complete; last status: $STATUS" >&2
  echo "$STATUS_JSON" | pretty
  exit 1
fi

echo
echo "→ /tasks/$TASK_ID/result"
curl -fsS "http://$HOST/tasks/$TASK_ID/result" | pretty
