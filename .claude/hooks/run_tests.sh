#!/usr/bin/env bash
# Stop hook: run the fast test tier, tests/runtests.jl (unit tests, about 2 minutes), when src/,
# tests/ or utils/posteriorHelpers.py differ from HEAD
# and from the last state that passed. Runs async; on failure it exits 2, which wakes Claude
# with the failing output. The regression and benchmark parts stay opt-in (env guards).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 0

input=$(cat)
active=$(printf '%s' "$input" | python3 -c 'import json,sys
try: print(str(json.load(sys.stdin).get("stop_hook_active", False)).lower())
except Exception: print("false")')

# Fingerprint of uncommitted changes the tests cover: src/, tests/ and the Python loaders that
# tests/derived_mode.jl exercises (tracked diff + untracked files).
changes=$( { git diff HEAD -- src tests utils/posteriorHelpers.py; \
             git ls-files --others --exclude-standard -- 'src/*.jl' 'tests/*.jl' 'tests/fixtures/*' | sort | xargs -r -d '\n' cat; } )
[ -z "$changes" ] && exit 0
hash=$(printf '%s' "$changes" | sha1sum | cut -d' ' -f1)

state=.claude/hooks/.last_pass
[ -f "$state" ] && [ "$(cat "$state")" = "$hash" ] && exit 0

log=$(mktemp)
if timeout 600 julia --project tests/runtests.jl >"$log" 2>&1; then
    echo "$hash" > "$state"
    rm -f "$log"
    exit 0
fi

summary=$(grep -E "Test Failed|Error|Evaluated|did not pass" "$log" | head -40)
tail_out=$(tail -30 "$log")
rm -f "$log"
msg="Unit tests FAILED (julia --project tests/runtests.jl) on the current uncommitted changes.
$summary
--- last lines ---
$tail_out"

if [ "$active" = "true" ]; then
    # Already re-woken once for this stop; report without looping.
    printf '%s\n' "$msg" >&2
    exit 1
fi
printf '%s\n' "$msg" >&2
exit 2
