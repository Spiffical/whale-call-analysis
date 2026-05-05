# Negative Manifest Dry Run

- Combined input rows: `1632`.
- Negative manifest rows: `1325`.
- ONC negative review queue rows: `925`.
- ONC reviewed-background rows: `0`.

## Negative Buckets

- `primary_adjacent_gap`: `825`
- `nonprimary_biological_signal`: `151`
- `nonbiological_signal`: `149`
- `external_source_gap`: `100`
- `ambiguous_hard_negative`: `100`

## Source Counts

- `DCLDE` combined input rows: `600`
- `ONC` combined input rows: `582`
- `BioDCASE` combined input rows: `450`

## Decision

- Do not launch GPU training from this dry run.
- The ONC reviewed-background bucket is still empty, so deployment background calibration remains blocked.
- Use `tables/dry_run_onc_negative_review_queue.csv` for the next human/visual review pass.
