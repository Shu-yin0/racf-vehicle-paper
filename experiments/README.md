# Experiment entry points

These scripts are the paper-facing analysis layer. They assume the corresponding local result/data manifests are present. They do not download assets or contact external services.

- `analyze_safety.py`: summarize paired safety outcomes.
- `analyze_conditional_coverage.py`: inspect conditional coverage summaries.
- `analyze_pathwise_results.py`: analyze pathwise result records.
- `analyze_pathwise_confirmatory_v3.py`: analyze the confirmatory protocol.
- `plot_safety.py` and `plot_coverage.py`: regenerate paper-facing plots.
- `generate_table.py`: generate compact LaTeX table fragments.
- `preflight_submission.py`: run submission/package consistency checks.

The frozen protocol is `pathwise_confirmatory_protocol_v3.yaml`. Scheduler launchers, internal host paths, and historical worker logs are intentionally excluded.
