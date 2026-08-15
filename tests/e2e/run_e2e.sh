#!/usr/bin/env bash
# WI-ACC-7: E2E shell test for Audio Chronicle Accelerator
#
# Usage:
#   ACCELERATOR_URL=http://localhost:8765 bash tests/e2e/run_e2e.sh
#
# Optional env vars:
#   ACCELERATOR_TOKEN  — bearer token (if server requires auth)
#   FIXTURE_PATH       — path to local audio file (skip download)
#   FIXTURE_URL        — URL to download the synthetic audio fixture from
#   FIXTURE_TOKEN      — optional auth token if FIXTURE_URL needs one
#   POLL_TIMEOUT       — max seconds to wait for job completion (default: 600)
#
# Exits 0 on pass, 1 on fail, 2 on skip (no fixture available).

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACCELERATOR_URL="${ACCELERATOR_URL:-http://localhost:8765}"
ACCELERATOR_TOKEN="${ACCELERATOR_TOKEN:-}"
POLL_TIMEOUT="${POLL_TIMEOUT:-600}"
POLL_INTERVAL=5
MIN_OVERLAP=0.80

# Optional URL to download the synthetic audio fixture from. Empty by default so
# the public suite never depends on an external host — provide a local file via
# FIXTURE_PATH, or set FIXTURE_URL to your own copy (FIXTURE_TOKEN if protected).
FIXTURE_DOWNLOAD_URL="${FIXTURE_URL:-}"

# Reference transcript (from annotated_segments.json)
REFERENCE_TEXT="Hey yeah sure So the plan for today is to finish up the speaker ID module Yeah totally agree I was thinking we should double check the resemblyzer version compatibility Good call I will add a smoke test for the VoiceEncoder import Okay so the afternoon session We are going to review the WI5 code The sidecar approach makes sense We might want a sanity check"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }
pass() { echo "[PASS] $*"; }
skip() { echo "[SKIP] $*" >&2; exit 2; }

auth_header() {
    if [[ -n "$ACCELERATOR_TOKEN" ]]; then
        echo "-H Authorization: Bearer $ACCELERATOR_TOKEN"
    fi
}

curl_authed() {
    if [[ -n "$ACCELERATOR_TOKEN" ]]; then
        curl -s -H "Authorization: Bearer $ACCELERATOR_TOKEN" "$@"
    else
        curl -s "$@"
    fi
}

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

log "Target: $ACCELERATOR_URL"

# Check required tools
for cmd in curl python3; do
    command -v "$cmd" >/dev/null 2>&1 || fail "Required tool not found: $cmd"
done

# Health check
log "Checking server health..."
HEALTH=$(curl_authed --max-time 10 "$ACCELERATOR_URL/health" 2>/dev/null || true)
if [[ -z "$HEALTH" ]]; then
    fail "Server not reachable at $ACCELERATOR_URL"
fi
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
if [[ "$STATUS" != "healthy" ]]; then
    fail "Health check failed: $HEALTH"
fi
log "Server is healthy."

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ -n "${FIXTURE_PATH:-}" && -f "$FIXTURE_PATH" ]]; then
    log "Using local fixture: $FIXTURE_PATH"
    AUDIO_FILE="$FIXTURE_PATH"
    AUDIO_NAME=$(basename "$AUDIO_FILE")
elif [[ -n "$FIXTURE_DOWNLOAD_URL" ]]; then
    log "Downloading fixture..."
    OGG_FILE="$TMP_DIR/ground_truth.ogg"
    if [[ -n "${FIXTURE_TOKEN:-}" ]]; then
        curl -s -L -H "Authorization: token $FIXTURE_TOKEN" \
            -o "$OGG_FILE" "$FIXTURE_DOWNLOAD_URL"
    else
        curl -s -L -o "$OGG_FILE" "$FIXTURE_DOWNLOAD_URL"
    fi
    [[ -s "$OGG_FILE" ]] || fail "Fixture download failed or empty: $OGG_FILE"

    # Try converting to WAV (optional — server accepts .ogg)
    WAV_FILE="$TMP_DIR/ground_truth.wav"
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -i "$OGG_FILE" "$WAV_FILE" -loglevel error && \
            AUDIO_FILE="$WAV_FILE" && AUDIO_NAME="ground_truth.wav" || \
            AUDIO_FILE="$OGG_FILE" && AUDIO_NAME="ground_truth.ogg"
    else
        AUDIO_FILE="$OGG_FILE"
        AUDIO_NAME="ground_truth.ogg"
    fi
    log "Fixture ready: $AUDIO_FILE ($(wc -c < "$AUDIO_FILE") bytes)"
else
    skip "No audio fixture available: set FIXTURE_PATH to a local file or FIXTURE_URL to a downloadable synthetic fixture."
fi

# ---------------------------------------------------------------------------
# Helper: poll a job to completion
# ---------------------------------------------------------------------------

poll_job() {
    local JOB_ID="$1"
    local LABEL="${2:-job}"
    local DEADLINE=$(( $(date +%s) + POLL_TIMEOUT ))
    log "Polling $LABEL ($JOB_ID)..."
    while true; do
        NOW=$(date +%s)
        if (( NOW > DEADLINE )); then
            fail "$LABEL timed out after ${POLL_TIMEOUT}s"
        fi
        JOB_JSON=$(curl_authed --max-time 15 "$ACCELERATOR_URL/v1/jobs/$JOB_ID")
        JOB_STATUS=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
        case "$JOB_STATUS" in
            complete)
                log "$LABEL completed."
                echo "$JOB_JSON"
                return 0
                ;;
            failed|cancelled)
                ERROR=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null || true)
                fail "$LABEL $JOB_STATUS: $ERROR"
                ;;
            *)
                log "  status=$JOB_STATUS — waiting ${POLL_INTERVAL}s..."
                sleep "$POLL_INTERVAL"
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Test 1: Transcription
# ---------------------------------------------------------------------------

log "=== TEST 1: Transcription ==="
TRANS_RESP=$(curl_authed --max-time 60 -X POST \
    -F "file=@${AUDIO_FILE};filename=${AUDIO_NAME}" \
    "$ACCELERATOR_URL/v1/audio/transcriptions")
TRANS_STATUS_CODE=$(echo "$TRANS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)

if [[ "$(echo "$TRANS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(str(d.get('cache_hit',False)).lower())" 2>/dev/null)" == "true" ]]; then
    log "Cache hit — result inline."
    TRANS_RESULT="$TRANS_RESP"
else
    TRANS_JOB_ID=$(echo "$TRANS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || true)
    [[ -n "$TRANS_JOB_ID" ]] || fail "No job_id in transcription response: $TRANS_RESP"
    TRANS_RESULT=$(poll_job "$TRANS_JOB_ID" "transcription")
fi

PREDICTED_TEXT=$(echo "$TRANS_RESULT" | python3 -c "
import sys,json
d=json.load(sys.stdin)
result = d.get('result') or d
print(result.get('text',''))
" 2>/dev/null || true)

log "Predicted text: ${PREDICTED_TEXT:0:120}..."

# Word overlap check
OVERLAP=$(python3 - <<EOF
predicted = """$PREDICTED_TEXT""".lower()
reference = """$REFERENCE_TEXT""".lower()
pred_words = set(predicted.split())
ref_words = reference.split()
if not ref_words:
    print("1.0")
else:
    matched = sum(1 for w in ref_words if w in pred_words)
    print(f"{matched / len(ref_words):.3f}")
EOF
)

log "Word overlap: $OVERLAP (threshold: $MIN_OVERLAP)"
PASS=$(python3 -c "print('yes' if float('$OVERLAP') >= float('$MIN_OVERLAP') else 'no')")
if [[ "$PASS" == "yes" ]]; then
    pass "Transcription word overlap: $OVERLAP ≥ $MIN_OVERLAP"
else
    fail "Transcription word overlap $OVERLAP below threshold $MIN_OVERLAP"
fi

# ---------------------------------------------------------------------------
# Test 2: Diarization
# ---------------------------------------------------------------------------

log "=== TEST 2: Diarization ==="
DIAR_RESP=$(curl_authed --max-time 60 -X POST \
    -F "file=@${AUDIO_FILE};filename=${AUDIO_NAME}" \
    "$ACCELERATOR_URL/v1/diarize")

if [[ "$(echo "$DIAR_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(str(d.get('cache_hit',False)).lower())" 2>/dev/null)" == "true" ]]; then
    log "Cache hit — result inline."
    DIAR_RESULT="$DIAR_RESP"
else
    DIAR_JOB_ID=$(echo "$DIAR_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || true)
    [[ -n "$DIAR_JOB_ID" ]] || fail "No job_id in diarization response: $DIAR_RESP"
    DIAR_RESULT=$(poll_job "$DIAR_JOB_ID" "diarization")
fi

SEGMENT_COUNT=$(echo "$DIAR_RESULT" | python3 -c "
import sys,json
d=json.load(sys.stdin)
result = d.get('result') or d
segs = result.get('segments', [])
print(len(segs))
" 2>/dev/null || true)

log "Diarization segments: $SEGMENT_COUNT"
if (( SEGMENT_COUNT >= 1 )); then
    pass "Diarization returned $SEGMENT_COUNT segment(s) ≥ 1"
else
    fail "Diarization returned $SEGMENT_COUNT segments (expected ≥ 1)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log ""
log "=== ALL E2E TESTS PASSED ==="
exit 0
