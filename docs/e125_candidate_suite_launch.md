# E125 Candidate Suite Launch Notes

E125 is the practical launcher for the next production-metric push. It runs the
small pairwise and two-stage experiments around an existing multiclass base run,
then builds an E124 leaderboard using the same production-style accounting:
cross-species false positives count against the predicted species and the true
species, while ONC background false positives remain visible.

## Why Two Data Variants Matter

Prior mixed-source runs showed the main risk clearly: BioDCASE/DCLDE can improve
some species sensitivity, but they can also damage ONC background rejection. E125
therefore supports two comparable variants from the same launch script:

- `ONConly`: filter the source manifest with `--source-kind ONC`.
- `full`: use every source in the manifest, including ONC plus external rows.

Use the same base run list and same held-out ONC reports when comparing them.

## ONC-Only Variant

```bash
cd /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/repo_e24_expert_hparam_68be99f
git pull

bash drac/scripts/submit_multispecies_e125_candidate_suite.sh \
  --variant-tag ONConly \
  --source-kind ONC \
  --source-manifest /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/manifests/e100_onc_only_blocked_nov_validation_20260611T020900Z/E101_stage2_ONConly_blocked_nov20_25_30_val/standardized_manifest.csv \
  --base-run-dir BASE_MULTICLASS_RUN_DIR
```

## Full-Data Variant

Point `FULL_STANDARDIZED_MANIFEST` at the standardized manifest that includes
ONC, BioDCASE, and DCLDE rows in the same label vocabulary.

```bash
cd /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/repo_e24_expert_hparam_68be99f
git pull

bash drac/scripts/submit_multispecies_e125_candidate_suite.sh \
  --variant-tag full \
  --source-manifest FULL_STANDARDIZED_MANIFEST \
  --base-run-dir BASE_MULTICLASS_RUN_DIR
```

Do not pass `--source-kind` for the full-data variant unless intentionally
running a source ablation.

## Monitoring

The submitter prints the suite directory. Monitor it with:

```bash
bash drac/scripts/monitor_multispecies_e125_candidate_suite.sh \
  --suite-dir SUITE_DIR \
  --show-disk
```

The monitor reports Slurm state, recent logs, report artifact presence, and the
first rows of the E124 leaderboard once complete. The finished suite should also
include `e124_candidate_examples.csv`; inspect that file before accepting a
winner so production false positives and cross-species errors are visible as
actual rows, not only counts.
