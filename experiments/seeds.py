"""Curated seed sets for variance testing.

Sequential seeds (1000, 1001, 1002) tend to cluster in LLM output
space — consecutive RNG states land in adjacent sampling trajectories.
Using seeds spread across the int32 range gives real variety.

The same seed set is used across variants so variance comparisons are
apples-to-apples (V1 seed 42 == V3 seed 42 at the LLM). Only the
pipeline shape differs.

Expand the set if a specific prompt consistently shows low variance —
could mean the seeds happen to land on similar trajectories for that
source. Reproducibility matters more than diversity for per-run
debugging, so the set is fixed, not randomly regenerated each session.
"""

# 5 seeds — default for variance testing. Chosen to span the int32 range.
SEEDS_5 = (
    42,
    137,
    7919,
    10_001,
    2_147_483_000,
)

# 10 seeds — use when 5 is too noisy to conclude.
SEEDS_10 = (
    42,
    137,
    1_729,
    7_919,
    10_001,
    99_991,
    1_000_003,
    65_537,
    524_287,
    2_147_483_000,
)

# 3 seeds — fast smoke checks where variance matters less than wall time.
SEEDS_3 = (
    42,
    7_919,
    10_001,
)
