# Reproduction guide

This package separates a public smoke gate from the paper's larger evidence runs.

## Public smoke gate

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python scripts/reproduce.py
```

The command creates:

- `results/smoke/coverage.json`: deterministic one-step and finite-horizon coverage summary;
- `results/smoke/coverage.png`: a small calibration diagnostic;
- `results/smoke/session_info.txt`: Python/platform information for the run.

The smoke gate uses synthetic transitions with seed `2026`. It is labelled `mock test` and does not claim to reproduce the paper's closed-loop benchmark.

## Paper result path

The paper-facing analysis scripts in `experiments/` operate on local protocol/data/model manifests. The initial private release includes compact summaries and manuscript assets; large raw arrays, policy checkpoints, worker databases, and scheduler records remain outside GitHub. To reproduce a paper number, obtain the matching manifest and place the assets at the paths documented in `data/README.md` and `models/README.md`, then record the command, seed, commit, and output hash.

## Figure/table crosswalk

| Paper item | Reference output | Public source or note |
|---|---|---|
| Fig. 2 evidence | `paper/figures/fig2_evidence.png` | source figure asset retained; generator snapshot: `experiments/gen_fig2_evidence.py` |
| Fig. 2 policy families | `paper/figures/fig2_policy_families.png` | source figure asset retained; generator snapshot: `experiments/gen_fig2_families.py` |
| Fig. 3 trade-offs | `paper/figures/fig3_tradeoffs.png` | source figure asset retained; generator snapshot: `experiments/gen_fig3_tradeoffs.py` |
| Safety summary | `results/safety_followup_v2_analysis.json` | compact analysis summary |
| Coverage summary | `results/conditional_coverage_v17.json` | compact analysis summary |
| Manuscript tables | `paper/generated/*.tex` | included LaTeX fragments |

The paper source and assets are checked in under `paper/` so the manuscript and repository can be audited together.

## Release audit

Before public release, verify that every reported estimate maps to a script, result file, or figure; rerun the smoke gate from a clean checkout; regenerate `MANIFEST-SHA256.txt`; and review all data/model licenses and author metadata.

