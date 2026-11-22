# SeqTrainer: instructions for coding agents

This file is the repository-wide operating guide for Codex and other coding agents. Read it before inspecting or changing code. Follow the most specific project documentation for the subsystem being changed.

## Project purpose

SeqTrainer is a Python toolkit for synthetic-biology and DNA-sequence machine-learning workflows. It connects SBOL and SynBioHub data to reproducible datasets, DNA-aware transforms, model integrations, and task-oriented applications. It complements Keras, PyTorch, and model ecosystems rather than replacing them.

The repository also contains research implementations and experimental workflows for bacterial DNA language modelling, including the Titans/MIRAS Memory-as-Context (MAC) work. Treat research code, stable library APIs, notebooks, and generated evidence as distinct deliverables.

## First steps

Before making changes:

1. Read this file and the root `README.md`.
2. Read `README_VISION.md` and `docs/architecture.md` when changing package boundaries or public APIs.
3. Inspect `pyproject.toml`, `.python-version`, and the relevant tests before changing dependencies or runtime behavior.
4. Read the closest relevant workflow document before touching a research workflow:
   - Bacterial dataset and legacy Titan MAC workflow: `docs/bacteria_titan.md`.
   - Titans Stage A: `docs/titans_stage_a/`.
   - Titans Stage B: `docs/titans_stage_b/`.
   - Titans Stage C: `docs/titans_stage_c/README.md`, `docs/titans_stage_c/CODEX_HANDOFF.md`, and the relevant handoff under `docs/titans_stage_c/handoffs/`.
5. Inspect the current branch, working tree, recent commits, and existing implementation. Do not assume that an old handoff or issue description still describes the current code.

Do not start expensive training, download large datasets, launch cloud/GPU jobs, or overwrite research artifacts as part of initial investigation.

## Repository map

- `src/seqtrainer/clients/`: remote clients, including SynBioHub.
- `src/seqtrainer/sparql/`: SPARQL prefixes, builders, and query recipes.
- `src/seqtrainer/data/`: SBOL loading, dataset recipes/materialization, and bacterial Titan data interfaces.
- `src/seqtrainer/transforms/`: DNA transforms and feature extraction.
- `src/seqtrainer/models/`: framework-neutral model metadata, registries, and retrieval helpers.
- `src/seqtrainer/keras/`: optional Keras adapters and factories.
- `src/seqtrainer/torch/`: optional PyTorch adapters, backbones, fine-tuning, and research model implementations.
- `src/seqtrainer/graph/`: RDF/SBOL graph utilities.
- `src/seqtrainer/applications/`: task-oriented application blueprints.
- `src/seqtrainer/cli/`: command-line interface.
- `tests/`: automated tests.
- `notebooks/tutorials/`: user-facing, runnable learning workflows.
- `scripts/`: operational, dataset-building, Colab, and research scripts.
- `docs/`: architecture, workflow contracts, research protocols, and handoffs.
- `artifacts/`: generated evidence and research deliverables. Treat committed artifacts as evidence, not disposable scratch space.
- `data/`: sample and project datasets. Check size, provenance, licensing, and tracking before changing or adding data.

This map is a guide, not permission to reorganize directories. Preserve established paths unless the task explicitly requires a migration.

## Architecture rules

- Keep biological and SBOL/SynBioHub domain semantics in framework-neutral modules. Do not move domain logic into Keras or PyTorch adapters.
- Keep optional frameworks optional. Base imports and base installation must not require TensorFlow, PyTorch, Transformers, or PyTorch Geometric unless the relevant feature explicitly requires them.
- Use lazy imports at optional integration boundaries. Missing optional dependencies should produce a clear, actionable error when that feature is invoked, not break unrelated imports.
- Keep stable public APIs under `src/seqtrainer/`. Research prototypes may remain isolated in experimental or clearly named research modules until their interfaces and tests justify promotion.
- Add task workflows through application-level composition of data recipes, transforms, models, and adapters. Avoid duplicating core functionality in notebooks or scripts.
- Preserve backward compatibility where practical. If a public API or serialized format must change, document the migration and update tests and examples.
- Do not introduce a new abstraction, dependency, framework, or directory layer when a small extension to the existing design is sufficient.

## Scientific and data integrity

Scientific correctness takes priority over attractive metrics or faster execution.

- Never fabricate, infer, or manually edit experimental results, benchmark values, hardware measurements, or success claims.
- Clearly distinguish measured results, synthetic smoke tests, estimates, hypotheses, and planned experiments.
- Do not describe a synthetic benchmark as evidence of biological performance.
- Preserve data provenance. Record source, version/release, retrieval or build procedure, preprocessing, split policy, random seeds, checksums, and relevant software/runtime versions where the workflow supports them.
- Make dataset construction and splitting deterministic when the scientific protocol requires reproducibility.
- Prevent leakage at the biologically meaningful grouping level. For bacterial sequence work, follow the relevant protocol's accession, genome, contig, species, ANI, or clade constraints. Never replace a documented group-aware split with random window splitting for convenience.
- Fit preprocessing, feature selection, normalization statistics, calibration, and decision thresholds on training/validation data only, as appropriate. Keep held-out test data for final evaluation.
- Report the actual evaluation unit and denominators. Mask padding and invalid positions from losses and metrics.
- Keep baseline comparisons fair: matched data, split, evaluation, and clearly documented differences in compute, parameter count, initialization, or training budget.
- Report uncertainty and per-group performance when the protocol requires it. Do not hide failed runs or selectively report favorable seeds.
- Store machine-readable manifests and raw metrics alongside summaries whenever the workflow defines them.

## Titans and bacterial DNA research safeguards

The Titans/MIRAS work has multiple generations. Do not conflate them.

- `src/seqtrainer/torch/titans_mac/` and `scripts/train_titan_mac_dna_lm_colab.py` are the legacy slot/EMA memory implementation and must remain identifiable as a baseline.
- `src/seqtrainer/torch/titans_paper_mac/`, `titans_paper_mac_stage_b/`, and `titans_paper_mac_stage_c/` are separate paper-MAC research paths. Do not silently replace, rename, or mutate the legacy implementation when extending paper-MAC.
- Treat Stage A/B audits, parity evidence, backend decisions, and Stage C handoffs as constraints. Read the current audit and relevant handoff before changing these paths.
- For Stage C, preserve the documented 32-token segment contract, clade/group-separated splits, stream-local memory lifecycle, and explicit gradient-horizon policy. Memory must not leak across unrelated streams or dataset splits.
- Keep correctness and fidelity gates ahead of performance optimization. Do not enable approximate scans, Flash attention, reduced precision, or another backend as a default unless the documented parity, causality, gradient, and evaluation gates pass.
- Do not claim a biological advantage from synthetic recall tests, short smoke runs, or hardware throughput alone.
- Do not launch a full-corpus run or exceed a documented compute/storage budget without explicit project-owner approval. Prefer bounded pilots and measured capacity estimates first.
- Checkpoint/resume behavior, random state, data cursor, stream identity/state, model configuration, and runtime metadata are part of reproducibility when relevant. Do not weaken checkpoint integrity to make a run appear successful.
- Keep Colab paths portable and parameterized. Do not introduce machine-specific absolute paths from a developer's local environment into shared code or documentation.

## Python and dependency conventions

- The repository pins Python 3.12 in `.python-version`. Check compatibility before using newer language or library features.
- Use the existing `pyproject.toml` and `uv.lock` as the dependency source of truth for the modern development workflow.
- Prefer `uv` for reproducible local development. Do not casually regenerate or hand-edit `uv.lock`.
- Keep dependency additions minimal and place framework-specific dependencies in the appropriate optional extra.
- Preserve the package's existing naming, typing, and import conventions. Use clear names and small, composable functions.
- Avoid broad exception handling that hides failures. Raise errors with context and preserve the original exception where appropriate.
- Do not add secrets, API tokens, private credentials, or personally identifying data to source, notebooks, fixtures, logs, or artifacts.

## Setup and validation

The documented development setup is:

```bash
uv sync --extra torch --extra dev
```

Run the test suite:

```bash
uv run pytest -q
```

Run focused tests while iterating, then the full suite for changes that affect shared code:

```bash
uv run pytest tests/<relevant_test_file>.py -q
uv run pytest -q
```

Run Ruff when changing Python code if it is available in the development environment:

```bash
uv run ruff check src tests
```

For changes involving optional integrations, validate both sides where practical:
- Base imports without the optional dependency installed.
- Feature-specific behavior with the relevant extra installed, or a clear dependency-gated test when installation is unavailable.

For changes to notebooks or scripts, validate syntax and the documented invocation. Do not execute a costly notebook or full training run merely to check syntax.

If a command cannot run because a dependency, GPU, dataset, credential, or external service is unavailable, report the exact limitation. Do not claim that validation passed.

## Tests and implementation expectations

- Add or update tests for changed behavior. Tests should be deterministic, focused, and runnable without network access or a GPU unless the test explicitly targets an integration.
- Use small synthetic fixtures for unit and smoke tests. Label synthetic data clearly and never use its scores as scientific results.
- Test optional-dependency failure paths and lazy imports.
- For data code, test schema, ordering, split boundaries, missing values, determinism, and provenance behavior where applicable.
- For model code, test shapes, masks, causality, gradients, state reset, checkpoint/resume, and numerical tolerances as relevant to the implementation.
- For CLI changes, test argument parsing, validation, output paths, and failure messages.
- Never weaken, delete, or loosen a test only to make a change pass. If a test is obsolete, explain why and replace the coverage.
- Do not update snapshots or expected scientific outputs without understanding and documenting why the underlying behavior should change.

## Notebooks, scripts, data, and artifacts

- Keep notebooks instructional and reproducible. Prefer calling package APIs over embedding a second implementation in notebook cells.
- Keep notebooks aligned with the current package API and document optional dependencies, hardware expectations, dataset inputs, and expected outputs.
- Keep scripts thin where possible. Shared logic belongs in importable package modules with tests.
- Do not commit generated caches, temporary checkpoints, raw downloads, huge intermediate arrays, or duplicate datasets unless the task explicitly requires a versioned artifact.
- Before adding or modifying a large file, inspect its size, provenance, license, and whether Git LFS or external storage is appropriate.
- Never overwrite a prior scientific artifact or checkpoint silently. Use a new run/output directory or obtain explicit approval.
- Preserve checksums and manifests. A report should be traceable to the exact code revision, configuration, dataset, and run that produced it.

## Git, branches, and pull requests

- Never commit feature work directly to `main`.
- Start from the intended base branch and create a focused branch with a descriptive name.
- Inspect `git status` before editing. Do not discard, reset, stash, or overwrite user changes.
- Keep changes scoped to the requested task. Avoid unrelated refactors and generated-file churn.
- Prefer small commits with one logical purpose and imperative commit messages.
- Do not force-push. Obtain explicit user permission before pushing branches or opening/updating a pull request.
- Do not merge a pull request unless the user explicitly asks.
- Summarize changed files, design decisions, tests run, tests not run, and any remaining risks in the final handoff.

## Documentation and communication

- Write concise, technically precise English.
- Preserve established biological, SBOL, model, and research terminology.
- Document user-visible API changes, configuration options, data contracts, reproducibility requirements, and migration steps.
- Do not state that a model is state-of-the-art, biologically validated, production-ready, or superior unless repository evidence supports the claim.
- Link to maintained repository documentation rather than copying long protocol descriptions into code comments.
- Comments should explain non-obvious constraints or scientific rationale, not narrate the code.

## Definition of done

A change is complete only when:

1. It addresses the requested task without unrelated scope.
2. The implementation follows the existing architecture and preserves relevant compatibility.
3. Tests cover the changed behavior and the applicable test suite passes, or limitations are clearly reported.
4. Scientific and data-integrity constraints remain intact.
5. Documentation, examples, and manifests are updated where needed.
6. The final diff contains no accidental datasets, secrets, checkpoints, caches, or unrelated generated files.
7. The handoff states what changed, why, validation evidence, and what remains unverified.
