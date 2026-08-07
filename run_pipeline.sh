#!/usr/bin/env bash
#
# PrediCT v2 Phase 2 — end-to-end pipeline runner.
#
# Runs stages 1 through 7 in order, with optional exploratory
# experiments at the end. Each stage logs to its own file under
# outputs/_logs/, with the master log at outputs/_logs/pipeline.log.
#
# Usage:
#   ./run_pipeline.sh                              # run everything
#   ./run_pipeline.sh --from 5                     # start at stage 5
#   ./run_pipeline.sh --from 5 --to 7              # 5, 6, 7 only
#   ./run_pipeline.sh --skip-exploratory           # skip the lesion experiment
#   ./run_pipeline.sh --n-jobs 40                  # override parallelism
#   ./run_pipeline.sh --dry-run                    # print what would run
#
# Exits non-zero on the first failing stage so CI / cron can detect it.

set -euo pipefail

# ─────────────────────── defaults ───────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_JOBS=80
STAGE_FROM=1
STAGE_TO=8
RUN_EXPLORATORY=true
DRY_RUN=false
CONDA_ENV="predict_env"

# ─────────────────────── CLI parsing ───────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)              STAGE_FROM="$2"; shift 2 ;;
        --to)                STAGE_TO="$2"; shift 2 ;;
        --n-jobs)            N_JOBS="$2"; shift 2 ;;
        --skip-exploratory)  RUN_EXPLORATORY=false; shift ;;
        --conda-env)         CONDA_ENV="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=true; shift ;;
        -h|--help)
            grep '^#' "$0" | head -25 | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $1"; exit 2 ;;
    esac
done

# ─────────────────────── activate conda env ───────────────────────

# Source the conda profile UNCONDITIONALLY. The parent shell may have
# `(predict_env)` in its prompt and CONDA_EXE in the env, but the `conda`
# shell function does NOT survive into a child shell — sub-bash starts
# cold and needs the profile script sourced even if the variables look set.
CONDA_PROFILE=""
for cand in \
    "/home/student/Student/projects/miniconda3" \
    "$HOME/miniconda3" \
    "/opt/miniconda3" \
    "${CONDA_PREFIX:-}/.." \
    "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"; do
    if [[ -n "$cand" && -f "$cand/etc/profile.d/conda.sh" ]]; then
        CONDA_PROFILE="$cand/etc/profile.d/conda.sh"
        break
    fi
done

if [[ -z "$CONDA_PROFILE" ]]; then
    echo "ERROR: could not find conda.sh profile script. Tried:"
    echo "  /home/student/Student/projects/miniconda3"
    echo "  \$HOME/miniconda3 = $HOME/miniconda3"
    echo "  /opt/miniconda3"
    echo "  \$CONDA_PREFIX/.. = ${CONDA_PREFIX:-unset}/.."
    echo "  \$CONDA_EXE-derived = ${CONDA_EXE:-unset}"
    echo "Edit run_pipeline.sh and add your conda root to the search loop."
    exit 1
fi

source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

# ─────────────────────── logging setup ───────────────────────

LOG_DIR="$PROJECT_ROOT/outputs/_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
MASTER_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

log() {
    local msg="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
    echo "$msg" | tee -a "$MASTER_LOG"
}

run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    local cmd="$3"
    local stage_log="$LOG_DIR/stage${stage_num}_${stage_name}_${TIMESTAMP}.log"

    if [[ "$stage_num" -lt "$STAGE_FROM" || "$stage_num" -gt "$STAGE_TO" ]]; then
        log "SKIP stage $stage_num ($stage_name) — outside [--from $STAGE_FROM, --to $STAGE_TO]"
        return 0
    fi

    log "STAGE $stage_num ($stage_name) START"
    log "  cmd : $cmd"
    log "  log : $stage_log"

    if $DRY_RUN; then
        log "  [DRY-RUN] not executing"
        return 0
    fi

    local t_start=$(date +%s)
    set +e
    bash -c "$cmd" >>"$stage_log" 2>&1
    local rc=$?
    set -e
    local t_end=$(date +%s)
    local dur=$((t_end - t_start))

    if [[ $rc -ne 0 ]]; then
        log "STAGE $stage_num ($stage_name) FAILED after ${dur}s (rc=$rc)"
        log "  tail of $stage_log:"
        tail -20 "$stage_log" | tee -a "$MASTER_LOG"
        exit "$rc"
    fi
    log "STAGE $stage_num ($stage_name) OK in ${dur}s"
}

# ─────────────────────── banner ───────────────────────

log "================================================================"
log " PrediCT v2 Phase 2 — pipeline run"
log "================================================================"
log " project root : $PROJECT_ROOT"
log " conda env    : $CONDA_ENV ($(python --version 2>&1))"
log " python path  : $(which python)"
log " stages       : $STAGE_FROM..$STAGE_TO"
log " --n-jobs     : $N_JOBS"
log " exploratory  : $RUN_EXPLORATORY"
log " dry run      : $DRY_RUN"
log " master log   : $MASTER_LOG"

# ─────────────────────── stages 1..4 (slow) ───────────────────────

run_stage 1 "io_discovery" \
    "python scripts/01_discover.py"

run_stage 2 "preprocess" \
    "python scripts/02_preprocess.py"

run_stage 3 "features" \
    "python scripts/03_features.py"

run_stage 4 "perturbations" \
    "python scripts/04_perturbations.py"

run_stage 4 "icc_gate" \
    "python scripts/05_icc_gate.py"

# ─────────────────────── stage 5 (reduce, ×3 cohorts) ───────────────────────

run_stage 5 "reduce_full" \
    "python scripts/06_reduce.py --n-jobs $N_JOBS"

run_stage 5 "reduce_Qr36d_2" \
    "python scripts/06_reduce.py --kernel-filter 'Qr36d/2' --n-jobs $N_JOBS"

run_stage 5 "reduce_I30f_3" \
    "python scripts/06_reduce.py --kernel-filter 'I30f/3' --n-jobs $N_JOBS"

# ─────────────────────── stage 6 (discover, ×3 cohorts) ───────────────────────

run_stage 6 "discover_full" \
    "python scripts/07_discover.py --cohort-dir outputs/06_reduce/ --n-jobs $N_JOBS"

run_stage 6 "discover_Qr36d_2" \
    "python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_Qr36d_2/ --n-jobs $N_JOBS"

run_stage 6 "discover_I30f_3" \
    "python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_I30f_3/ --n-jobs $N_JOBS"

# ─────────────────────── stage 7 (analyse) ───────────────────────

run_stage 7 "analyse" \
    "python scripts/08_analyse.py"

run_stage 7 "analyse_report" \
    "python scripts/08d_stage7_results.py | tee outputs/07_analyse/stage7_results_report.txt"

run_stage 7 "analyse_verify" \
    "python scripts/08e_verify_focal_diffuse.py | tee outputs/07_analyse/focal_diffuse_verification.txt"

# ─────────────────────── stage 8 (validate) ───────────────────────

run_stage 8 "validate" \
    "python scripts/09_validate.py"

# ─────────────────────── lesion-morphology exploratory ───────────────────────

if $RUN_EXPLORATORY && [[ "$STAGE_TO" -ge 7 ]]; then
    log "----------------------------------------------------------------"
    log " EXPLORATORY: lesion morphology"
    log "----------------------------------------------------------------"

    run_stage 7 "lesion_morph_run" \
        "python experiments/lesion_morphology/run.py --n-jobs $N_JOBS"

    run_stage 7 "lesion_morph_analyse" \
        "python experiments/lesion_morphology/analyse.py --n-jobs $N_JOBS"

    run_stage 7 "lesion_morph_finalise" \
        "python experiments/lesion_morphology/finalise.py --n-jobs $N_JOBS"
fi

# ─────────────────────── done ───────────────────────

log "================================================================"
log " PIPELINE COMPLETE"
log "================================================================"
log " all stages OK; outputs under $PROJECT_ROOT/outputs/"
log " master log: $MASTER_LOG"
log ""
log " quick spot-check commands:"
log "   ls outputs/01_manifest/manifest.csv"
log "   ls outputs/03_features/features.csv"
log "   ls outputs/05_icc/gated_features.csv"
log "   ls outputs/06_reduce/{representative_features,cohort_metadata,prepared_matrix}.csv"
log "   ls outputs/07_analyse/{phenotype_paper_table,directional_verdict.json,burden_orthogonality}.csv"
log "   cat outputs/07_analyse/directional_verdict.json"
log "   ls outputs/08_validate/{external_holdout_report,leave_k_out_ari,cross_cohort_ari_consolidated}.csv"
log "   cat outputs/08_validate/run_header_validate.json | python -m json.tool | head -40"
