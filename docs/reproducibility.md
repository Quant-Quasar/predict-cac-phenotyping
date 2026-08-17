# Reproducibility

## Determinism
- All randomised steps read `seed` from `configs/default.yaml` (default: 42).
- No module uses a hardcoded `random_state`. Seeds are passed explicitly.
- PyRadiomics is fully deterministic given the same `params.yaml` and input.

## Environment
- `pyproject.toml` declares the package.
- Install: `pip install -e .` from the repo root.
- Lock pinned dependencies via `pip freeze > requirements.lock` once the environment is finalised.

## Stage-by-stage outputs
- Every stage writes CSV/NPY at the boundary.
- Rerunning a single stage requires only the previous stage's outputs.
- File naming: `outputs/<NN_stage>/...`.

## Logs
- Each script writes a structured log to `outputs/logs/{script_name}.log`.
- Log includes: seed, config hash, input file hashes, output file hashes, runtime.

## Decision tracking
- One markdown per decision in `decisions/D###-slug.md`.
- Code that implements a decision cites it in the docstring header.
