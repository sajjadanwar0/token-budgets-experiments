"""recover_anthropic_partial.py

Reconstructs the anthropic trials you already ran from your stdout log,
so the new (resume-capable) m2_run.py picks up where you left off
without re-spending the ~$0.20 on the 30 coarse trials.

What this writes to m2_haiku_lang001_cap2000_n30.csv:

  Coarse trials (token_capabilities):  30 rows, COMPLETE and EXACT.
     Per-trial total_spent_uc taken directly from your stdout. All trials
     ran to recursion_limit (10 steps) and reported outcome
     'completed_no_cap_hit' with 0 overshoot.

  Bytelen trials (token_capabilities_bytelen): 18 rows, APPROXIMATE.
     spent_uc and overshoot exact from stdout; agent_steps is approximated
     as 4 (not captured in stdout). If you want exact agent_steps for these
     trials, DELETE them from the CSV before running m2_run.py - the resume
     logic will re-run them fresh. The headline 0-overshoot result is
     unchanged either way; only the steps column is at issue.

After running this:

  python3 m2_run.py --provider anthropic --cap-uc 2000 --runs 30 \\
      --output-csv m2_haiku_lang001_cap2000_n30.csv

  ... will resume by running:
    - 0  more token_capabilities (already complete)
    - 12 more token_capabilities_bytelen (trials 18-29)
    - 30 naive_guard (trials 0-29)

  Total remaining live-API cost: ~$0.30, ~10 min wall-clock.
"""
import csv

# From your stdout, token_capabilities trial 0..29
COARSE_SPENT = [
    1635, 1528, 1482, 1589, 1756, 1678, 1485, 1568, 1635, 1620,
    1681, 1494, 1681, 1557, 1654, 1662, 1518, 1667, 1594, 1540,
    1460, 1514, 1538, 1537, 1586, 1495, 1496, 1603, 1554, 1564,
]
# From your stdout, token_capabilities_bytelen trial 0..17
BYTELEN_SPENT = [
    497, 416, 488, 415, 475, 461, 493, 476, 363, 505,
    470, 416, 470, 449, 505, 500, 493, 510,
]
assert len(COARSE_SPENT) == 30
assert len(BYTELEN_SPENT) == 18

CAP = 2000
PROVIDER = "anthropic"
MODEL = "claude-haiku-4-5-20251001"
WORKLOAD = "lang001"

FIELDNAMES = [
    "runtime", "outcome", "agent_steps", "cap_uc", "total_spent_uc",
    "pct_of_cap", "overshoot_uc", "structural_undershoot_uc",
    "wasted_call_cost_uc", "provider", "model", "workload", "trial",
]

rows = []
for trial, spent in enumerate(COARSE_SPENT):
    rows.append({
        "runtime": "token_capabilities",
        "outcome": "completed_no_cap_hit",
        "agent_steps": 10,
        "cap_uc": CAP,
        "total_spent_uc": spent,
        "pct_of_cap": round(spent / CAP * 100, 2),
        "overshoot_uc": max(0, spent - CAP),
        "structural_undershoot_uc": 0,
        "wasted_call_cost_uc": 0,
        "provider": PROVIDER, "model": MODEL, "workload": WORKLOAD,
        "trial": trial,
    })

for trial, spent in enumerate(BYTELEN_SPENT):
    rows.append({
        "runtime": "token_capabilities_bytelen",
        "outcome": "compile_time_reservation_refused",
        "agent_steps": 4,   # APPROXIMATE - see header docstring
        "cap_uc": CAP,
        "total_spent_uc": spent,
        "pct_of_cap": round(spent / CAP * 100, 2),
        "overshoot_uc": max(0, spent - CAP),
        "structural_undershoot_uc": 0,
        "wasted_call_cost_uc": 0,
        "provider": PROVIDER, "model": MODEL, "workload": WORKLOAD,
        "trial": trial,
    })

with open("m2_haiku_lang001_cap2000_n30.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to m2_haiku_lang001_cap2000_n30.csv")
print(f"  - 30 token_capabilities (coarse, exact from stdout)")
print(f"  - 18 token_capabilities_bytelen (spent/overshoot exact; agent_steps approx.)")
print(f"  - 0  naive_guard (none yet; 30 to run)")
print()
print("Next step: re-run the same m2_run.py command. It will resume and")
print("only run the remaining 12 bytelen + 30 naive_guard trials.")