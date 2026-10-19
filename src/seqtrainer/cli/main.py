"""Command line interface for common SeqTrainer workflows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from seqtrainer._deprecation import warn_deprecated
from seqtrainer.applications.promoter_regression import build_promoter_regression_blueprint
from seqtrainer.data import list_builtin_dataset_recipes, materialize_dataset_from_sbol
from seqtrainer.data.sbol import get_sequence_from_sbol
from seqtrainer.keras.factories import create_keras_model
from seqtrainer.sparql.prefixes import format_prefixes
from seqtrainer.torch.finetune import build_finetune_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seqtrainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_sbol = subparsers.add_parser("inspect-sbol", help="Inspect one SBOL file")
    inspect_sbol.add_argument("file", type=Path)

    dataset = subparsers.add_parser("dataset", help="Dataset recipe and materialization commands")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    recipes_cmd = dataset_sub.add_parser("recipes", help="List built-in dataset recipes")
    recipes_cmd.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    build_dataset = dataset_sub.add_parser("build", help="Build dataset from local SBOL files")
    build_dataset.add_argument("files", nargs="+", type=Path)
    build_dataset.add_argument("--recipe", default="local-sbol-regression")
    build_dataset.add_argument(
        "--y-uri",
        default="http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue",
    )
    build_dataset.add_argument("--output", type=Path, help="Write dataset output to file")
    build_dataset.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    build_dataset.add_argument("--cache", action="store_true", help="Write dataset snapshot cache manifest")
    build_dataset.add_argument("--cache-dir", type=Path)
    build_dataset.add_argument("--dataset-name", default="sbol-local")
    build_dataset.add_argument("--dataset-version")

    legacy_build = subparsers.add_parser("build-dataset", help="(Deprecated) alias for dataset build")
    legacy_build.add_argument("files", nargs="+", type=Path)
    legacy_build.add_argument("--recipe", default="local-sbol-regression")
    legacy_build.add_argument("--y-uri", default="http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue")
    legacy_build.add_argument("--output", type=Path)
    legacy_build.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    legacy_build.add_argument("--cache", action="store_true")
    legacy_build.add_argument("--cache-dir", type=Path)
    legacy_build.add_argument("--dataset-name", default="sbol-local")
    legacy_build.add_argument("--dataset-version")

    sparql = subparsers.add_parser("sparql", help="SPARQL helpers")
    sparql_sub = sparql.add_subparsers(dest="sparql_command", required=True)
    sparql_sub.add_parser("prefixes", help="Print default prefixes")

    model = subparsers.add_parser("model", help="Framework-specific model build commands")
    model_sub = model.add_subparsers(dest="model_command", required=True)

    model_build = model_sub.add_parser("build", help="Build framework model/fine-tune config")
    model_build.add_argument("--framework", choices=["torch", "keras"], default="torch")
    model_build.add_argument("--task", choices=["regression", "classification"], default="regression")
    model_build.add_argument("--backbone", default="dnabert2")
    model_build.add_argument("--head", default="regression-mlp")
    model_build.add_argument("--learning-rate", type=float, default=1e-4)
    model_build.add_argument("--epochs", type=int, default=5)
    model_build.add_argument("--output", type=Path)

    benchmark = subparsers.add_parser("benchmark", help="Reproducible benchmark commands")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    for name, help_text in (("run", "Run a configured benchmark"), ("manifest", "Validate config and write a manifest")):
        command = benchmark_sub.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--base-dir", type=Path, default=Path.cwd())
        if name == "run":
            command.add_argument("--strict", action="store_true", help="Fail instead of recording an opt-in skip")
    compare = benchmark_sub.add_parser("compare", help="Compare compatible completed manifests")
    compare.add_argument("artifact_dirs", nargs="+", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    for name, help_text in (("prepare-dnabert2", "Prepare offline DNABERT2 tokenized splits"), ("prepare-ipromp", "Prepare FASTA and mapping files for external iPro-MP")):
        command = benchmark_sub.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--base-dir", type=Path, default=Path.cwd())

    annotate = subparsers.add_parser("annotate", help="Annotation workflows")
    annotate_sub = annotate.add_subparsers(dest="annotate_command", required=True)
    annotate_promoters = annotate_sub.add_parser(
        "promoters", help="Annotate predicted promoters in GenBank files"
    )
    annotate_promoters.add_argument("input", type=Path)
    annotate_promoters.add_argument("--model-family", choices=("dnabert2", "dummy"), default="dummy")
    annotate_promoters.add_argument(
        "--model-bundle",
        type=Path,
        help="Folder containing a trained checkpoint and matching benchmark manifest",
    )
    annotate_promoters.add_argument("--checkpoint", type=Path)
    annotate_promoters.add_argument("--benchmark-manifest", type=Path)
    annotate_promoters.add_argument("--threshold", type=float)
    annotate_promoters.add_argument("--window-size", type=int)
    annotate_promoters.add_argument("--step-size", type=int, default=25)
    annotate_promoters.add_argument("--scan-both-strands", action=argparse.BooleanOptionalAction, default=True)
    annotate_promoters.add_argument("--merge-distance", type=int, default=25)
    annotate_promoters.add_argument("--predictions-csv", type=Path)
    annotate_promoters.add_argument("--manifest", type=Path)
    annotate_promoters.add_argument("--output", type=Path)
    annotate_promoters.add_argument("--evaluation-dir", type=Path)
    annotate_promoters.add_argument("--sbol-output", type=Path)
    annotate_promoters.add_argument(
        "--sbol2-output",
        type=Path,
        help="Write SBOL2 RDF/XML with Canvas-renderable feature components",
    )
    annotate_promoters.add_argument("--sbol-namespace", default="https://seqtrainer.org/designs")
    annotate_promoters.add_argument("--promoter-label-mode", choices=("strict", "labelled"), default="labelled")
    annotate_promoters.add_argument(
        "--annotation-completeness",
        choices=("verified_complete", "partial", "unknown"),
        default="unknown",
    )
    annotate_promoters.add_argument("--iou-threshold", type=float, default=0.50)
    annotate_promoters.add_argument("--clean-output", action="store_true")
    annotate_promoters.add_argument("--open-output-folder", action="store_true")

    annotate_collection = annotate_sub.add_parser(
        "promoter-collection", help="Evaluate a collection of labelled GenBank plasmids"
    )
    annotate_collection.add_argument("--manifest", type=Path, required=True)
    annotate_collection.add_argument("--input-dir", type=Path, required=True)
    annotate_collection.add_argument("--output-dir", type=Path, required=True)
    annotate_collection.add_argument("--model-family", choices=("dnabert2", "dummy"), default="dummy")
    annotate_collection.add_argument("--checkpoint", type=Path)
    annotate_collection.add_argument("--benchmark-manifest", type=Path)
    annotate_collection.add_argument("--threshold", type=float)
    annotate_collection.add_argument("--window-size", type=int)
    annotate_collection.add_argument("--sbol-namespace", default="https://seqtrainer.org/designs")
    annotate_collection.add_argument("--promoter-label-mode", choices=("strict", "labelled"), default="labelled")
    annotate_collection.add_argument(
        "--annotation-completeness",
        choices=("verified_complete", "partial", "unknown"),
        default="unknown",
    )
    annotate_collection.add_argument("--write-sbol3", action="store_true")
    annotate_collection.add_argument("--continue-on-error", action="store_true")

    return parser


def _write_dataset_output(dataset_rows: list[dict], *, output_path: Path | None, output_format: str) -> None:
    if output_format == "csv":
        import pandas as pd

        payload = pd.DataFrame(dataset_rows).to_csv(index=False)
    elif output_format == "jsonl":
        payload = "\n".join(json.dumps(row, sort_keys=True) for row in dataset_rows)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported output format: {output_format}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + ("\n" if not payload.endswith("\n") else ""), encoding="utf-8")
    else:
        print(payload)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-sbol":
        seq = get_sequence_from_sbol(args.file)
        print(f"sequence_length={len(seq) if seq else 0}")
        return 0

    if args.command == "dataset" and args.dataset_command == "recipes":
        recipes = list_builtin_dataset_recipes()
        if args.json:
            print(json.dumps({k: v.as_manifest_payload() for k, v in recipes.items()}, indent=2, sort_keys=True))
            return 0
        for name, recipe in recipes.items():
            print(f"{name}: sequence_field={recipe.sequence_field} label_field={recipe.label_field}")
        return 0

    if (args.command == "dataset" and args.dataset_command == "build") or args.command == "build-dataset":
        if args.command == "build-dataset":
            warn_deprecated(old="seqtrainer build-dataset", new="seqtrainer dataset build", kind="command", stacklevel=2)

        dataset, manifest = materialize_dataset_from_sbol(
            args.files,
            y_uri=args.y_uri,
            recipe_name=args.recipe,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            cache_dir=args.cache_dir,
            write_cache=args.cache,
        )
        _write_dataset_output(dataset.examples, output_path=args.output, output_format=args.output_format)
        if manifest:
            print(f"# cache_snapshot={manifest.snapshot_key}")
        return 0

    if args.command == "benchmark":
        if args.benchmark_command == "run":
            from seqtrainer.benchmarks import run_benchmark

            result = run_benchmark(args.config, base_dir=args.base_dir, output_dir=args.output_dir, allow_skip=not args.strict)
            print(f"status={result.status}\noutput_dir={result.output_dir}")
            return 0
        if args.benchmark_command == "manifest":
            return _write_benchmark_manifest(args.config, args.output_dir, args.base_dir)
        if args.benchmark_command == "compare":
            from seqtrainer.benchmarks import compare_benchmark_outputs

            written = compare_benchmark_outputs(args.artifact_dirs, output_dir=args.output_dir)
            print(f"comparison_metrics={written['comparison_metrics']}\ncomparison_summary={written['comparison_summary']}")
            return 0
        if args.benchmark_command == "prepare-dnabert2":
            from seqtrainer.benchmarks import prepare_dnabert2_tokenized_splits

            tokenized = prepare_dnabert2_tokenized_splits(args.config, base_dir=args.base_dir, output_dir=args.output_dir)
            print(f"output_dir={tokenized.output_dir}\nmetadata={tokenized.metadata_path}")
            return 0
        if args.benchmark_command == "prepare-ipromp":
            from seqtrainer.adapters.ipromp import prepare_ipromp_inputs

            prepared = prepare_ipromp_inputs(args.config, base_dir=args.base_dir, output_dir=args.output_dir)
            print(f"output_dir={prepared.output_dir}\nmapping_csv={prepared.mapping_csv}\ncommand_script={prepared.command_script}")
            return 0

    if args.command == "annotate":
        if args.annotate_command == "promoters":
            from seqtrainer.annotation import PromoterAnnotationConfig, run_promoter_annotation

            manifest = run_promoter_annotation(
                PromoterAnnotationConfig(
                    input_file=args.input,
                    output_file=args.output,
                    predictions_csv=args.predictions_csv,
                    manifest=args.manifest,
                    model_family=args.model_family,
                    model_bundle=args.model_bundle,
                    checkpoint=args.checkpoint,
                    benchmark_manifest=args.benchmark_manifest,
                    threshold=args.threshold,
                    window_size=args.window_size,
                    step_size=args.step_size,
                    scan_both_strands=args.scan_both_strands,
                    merge_distance=args.merge_distance,
                    evaluation_dir=args.evaluation_dir,
                    sbol_output=args.sbol_output,
                    sbol2_output=args.sbol2_output,
                    sbol_namespace=args.sbol_namespace,
                    promoter_label_mode=args.promoter_label_mode,
                    annotation_completeness=args.annotation_completeness,
                    iou_threshold=args.iou_threshold,
                    clean_output=args.clean_output,
                )
            )
            print(f"output_file={manifest['output_file']}")
            print(f"predictions_csv={manifest['predictions_csv']}")
            print(f"manifest={manifest['manifest_file']}")
            print(f"predicted_promoters_added={manifest['predicted_promoters_added']}")
            if args.open_output_folder:
                _open_output_folder(Path(manifest["output_file"]).parent)
            return 0

        if args.annotate_command == "promoter-collection":
            from seqtrainer.annotation.collection import run_promoter_collection

            result = run_promoter_collection(
                args.manifest,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_family=args.model_family,
                checkpoint=args.checkpoint,
                benchmark_manifest=args.benchmark_manifest,
                threshold=args.threshold,
                window_size=args.window_size,
                sbol_namespace=args.sbol_namespace,
                promoter_label_mode=args.promoter_label_mode,
                write_sbol3=args.write_sbol3,
                continue_on_error=args.continue_on_error,
                annotation_completeness=args.annotation_completeness,
            )
            print(f"collection_manifest={args.output_dir / 'collection_manifest.json'}")
            print(f"included_count={result['included_count']}")
            print(f"excluded_count={result['excluded_count']}")
            return 0

    if args.command == "sparql" and args.sparql_command == "prefixes":
        print(format_prefixes())
        return 0

    if args.command == "model" and args.model_command == "build":
        if args.framework == "torch":
            payload = {
                "framework": "torch",
                "model_blueprint": build_promoter_regression_blueprint("torch"),
                "finetune": build_finetune_config(
                    backbone=args.backbone,
                    head=args.head,
                    task=args.task,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                ),
            }
        else:
            payload = {
                "framework": "keras",
                "model_blueprint": build_promoter_regression_blueprint("keras"),
                "factory_output": create_keras_model(
                    backbone=args.backbone,
                    head=args.head,
                    task=args.task,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                ),
            }

        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    parser.error("Unhandled command")
    return 2


def _open_output_folder(path: Path) -> None:
    resolved = path.resolve()
    if os.name == "nt":
        getattr(os, "startfile")(str(resolved))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(resolved)], check=False)
    else:
        subprocess.run(["xdg-open", str(resolved)], check=False)


def _write_benchmark_manifest(config_path: Path, output_dir_arg: Path | None, base_dir: Path) -> int:
    from seqtrainer.benchmarks import (
        build_run_manifest,
        load_benchmark_config,
        load_predefined_split_frames,
        summarize_split_frames,
        write_benchmark_outputs,
    )

    config = load_benchmark_config(config_path)
    frames = load_predefined_split_frames(config, base_dir=base_dir)
    split_summary = summarize_split_frames(config, frames)
    manifest = build_run_manifest(config, repo_dir=base_dir, split_summary=split_summary)
    output_dir = output_dir_arg or Path(config.outputs.output_dir)
    write_benchmark_outputs(output_dir, manifest=manifest, config=config)
    print(f"output_dir={output_dir}\nmanifest={output_dir / 'manifest.json'}")
    for split, summary in split_summary.items():
        print(f"{split}: rows={summary['rows']} class_counts={summary['class_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
