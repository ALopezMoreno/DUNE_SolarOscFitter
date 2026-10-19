#=
Test tiers (CLAUDE.md "Which tests to run"):
  fast  (default)       unit tests, about 2 minutes: run on every change (the Stop hook does).
  slow  (JULIA_TESTS=slow)  + the Enzyme unit tests (compile ~1 min), the warm-start sampling tests
                        and derived mode end to end: run before handing a change over or merging.
JULIA_REGRESSION=1 adds the full-likelihood regression in either tier (~5 min): run it when the
change can move the likelihood value.
=#
const TEST_TIER = get(ENV, "JULIA_TESTS", "fast")
TEST_TIER in ("fast", "slow") || error("JULIA_TESTS must be \"fast\" or \"slow\", got \"$TEST_TIER\"")
const RUN_SLOW = TEST_TIER == "slow"
