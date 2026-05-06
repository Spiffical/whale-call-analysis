# ONC Negative Review Visual Sample

- Queue rows: `925`.
- Primary-adjacent gap sample: `32` rows, `32` rendered panels.
- Ambiguous hard-negative sample: `24` rows, `24` rendered panels.

## Decision

- Keep training blocked until enough primary-adjacent gaps are reviewed as clean `reviewed_background`.
- Ambiguous hard negatives remain useful for training pressure but must not define the deployment background gate.
