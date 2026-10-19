#!/usr/bin/env bash
# Run mork-train and restart it after a crash (e.g. a WSL "CUDA unknown error").
# Each restart resumes from latest.pt in the checkpoint dir, so at most
# checkpoint_every steps are lost per crash.
#
# Stops on exit 0 (run finished) or 130 (Ctrl-C, latest.pt saved), and when a
# run fails within MIN_RUNTIME seconds (a config error, not a mid-run crash).
#
#   scripts/train_loop.sh --max-steps 60884 --max-lr 2e-4
#   MAX_RESTARTS=5 scripts/train_loop.sh --checkpoint-dir checkpoints/cool_24k --run-name cool_24k
set -u

MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY="${RESTART_DELAY:-30}"
MIN_RUNTIME="${MIN_RUNTIME:-300}"

for arg in "$@"; do
    case "$arg" in
        --fresh|--no-resume)
            echo "train_loop: $arg would restart from scratch after every crash; run mork-train directly" >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/.."
trap 'exit 130' INT

restarts=0
while true; do
    started=$(date +%s)
    echo "train_loop: starting mork-train $* ($(date -Is))"
    uv run mork-train "$@"
    code=$?
    runtime=$(( $(date +%s) - started ))

    if [ "$code" -eq 0 ]; then
        echo "train_loop: finished"
        exit 0
    fi
    if [ "$code" -eq 130 ]; then
        echo "train_loop: interrupted, not restarting"
        exit 130
    fi
    if [ "$runtime" -lt "$MIN_RUNTIME" ]; then
        echo "train_loop: exit $code after ${runtime}s (< ${MIN_RUNTIME}s), not restarting" >&2
        exit "$code"
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "train_loop: exit $code, restart limit $MAX_RESTARTS reached" >&2
        exit "$code"
    fi

    restarts=$((restarts + 1))
    echo "train_loop: exit $code after ${runtime}s, restart $restarts/$MAX_RESTARTS in ${RESTART_DELAY}s ($(date -Is))" >&2
    sleep "$RESTART_DELAY"
done
