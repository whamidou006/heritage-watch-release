"""Command-line entry points for the three reproducible workflows."""
import argparse
import json
from pathlib import Path

from .config import load_config
from .features import REPRESENTATIONS, SELECTED, assemble, load_features
from .protocol import compare, evaluate, report


def reproduce(config, from_cache=None, device="cpu", batch_size=32):
    """Compute (never substitute published numbers for) the section-5 table."""
    from .encoders import embed_manifest
    from .manifest import build_manifest, print_manifest, write_manifest
    if from_cache is None:
        root = Path("cache")
        manifest = build_manifest(config)
        print_manifest(manifest)
        write_manifest(manifest, root / "manifest.json")
        for encoder, filename in (("dinov2", "features_month.npz"),
                                  ("satlas", "features_satlas_month.npz")):
            embed_manifest(config, manifest, encoder, root / filename, device, batch_size)
    else:
        root = Path(from_cache)
    blocks, meta = load_features([root / "features_month.npz",
                                 root / "features_satlas_month.npz"], config)
    results = {}
    print(f"n={len(meta['y'])}; paired jittered grids; CPU classifier; PCA off", flush=True)
    for name, rep in REPRESENTATIONS.items():
        r = evaluate(assemble(blocks, name), meta["y"], meta, **config.evaluation_kwargs())
        results[name] = r
        print(f"{rep.title:<42} {rep.dimension:5d} {r.macro_f1:.4f} sd={r.sd:.4f}", flush=True)
    for name, r in results.items():
        if name != SELECTED:
            compare(results[SELECTED], r, SELECTED, name, config.min_effect)
    return results


def parser():
    ap = argparse.ArgumentParser(description="Heritage Watch: frozen encoders and paired spatial change evaluation.")
    sub = ap.add_subparsers(dest="command", required=True)
    descriptions = {
        "manifest": "Resolve date pairs and print the usable-label/drop ledger.",
        "embed": "Extract full-window chips and cache frozen encoder features.",
        "evaluate": "Evaluate features with paired jittered spatial cross-validation.",
        "compare": "Compare two representations on identical grids using all three decision bars.",
        "train": "Fit the feature head on all labels and save a portable model bundle.",
        "predict": "Apply a trusted trained bundle to compatible new-site labelled points.",
        "reproduce": "Compute the seven section-5 representations (expensive; caches recommended).",
    }
    for name, description in descriptions.items():
        p = sub.add_parser(name, help=description, description=description)
        p.add_argument("--config", required=True, help="Site YAML; relative data paths resolve beside this file.")
        if name in ("evaluate", "compare", "train"):
            p.add_argument("--features", nargs="+", required=True,
                           help="One or both NPZ encoder caches; fid and all metadata must match in order.")
        if name in ("evaluate", "train"):
            p.add_argument("--representation", required=True, choices=REPRESENTATIONS,
                           help="Named representation assembled from the supplied encoder blocks.")
        if name == "compare":
            p.add_argument("--a", required=True, choices=REPRESENTATIONS, help="First representation; reports A minus B.")
            p.add_argument("--b", required=True, choices=REPRESENTATIONS, help="Second representation on identical rows/grids.")
        if name in ("manifest", "embed", "train", "predict"):
            p.add_argument("--out", required=name != "manifest", default="cache/manifest.json",
                           help={"manifest": "Output manifest JSON (default: cache/manifest.json).",
                                 "embed": "Output feature cache NPZ.",
                                 "train": "Output model bundle, e.g. out/model.joblib.",
                                 "predict": "Output CSV with fid, truth, prediction and class probabilities."}[name])
        if name == "embed":
            p.add_argument("--manifest", required=True, help="JSON produced by the manifest command.")
            p.add_argument("--encoder", required=True, choices=("dinov2", "satlas"),
                           help="Frozen encoder family; satlas writes SI t1/t2 and fused MI blocks.")
        if name == "predict":
            p.add_argument("--model", required=True, help="Trusted joblib training bundle; never load untrusted models.")
        if name == "reproduce":
            p.add_argument("--from-cache", help="Directory containing features_month.npz and features_satlas_month.npz; "
                           "omit to extract all features from scratch into cache/.")
        if name in ("embed", "predict", "reproduce"):
            p.add_argument("--device", default="cpu", help="Embedding device (default cpu); use cuda with CUDA_VISIBLE_DEVICES set.")
            p.add_argument("--batch-size", type=int, default=32, help="Encoder inference batch size (default 32).")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "manifest":
            from .manifest import build_manifest, print_manifest, write_manifest
            man = build_manifest(config)
            print_manifest(man)
            write_manifest(man, args.out)
        elif args.command == "embed":
            from .encoders import embed_manifest
            embed_manifest(config, json.loads(Path(args.manifest).read_text()), args.encoder,
                           args.out, args.device, args.batch_size)
        elif args.command == "predict":
            from .predict import predict
            predict(args.model, config, args.out, args.device, args.batch_size)
        elif args.command == "reproduce":
            reproduce(config, args.from_cache, args.device, args.batch_size)
        else:
            blocks, meta = load_features(args.features, config)
            if args.command == "train":
                from .train import train
                bundle = train(config, blocks, meta, args.representation, args.out)
                print(f"trained n={bundle['n_training']} dim={bundle['feature_dimension']} "
                      f"fingerprint={bundle['training_fingerprint']}")
            else:
                def score(name):
                    result = evaluate(assemble(blocks, name), meta["y"], meta,
                                      **config.evaluation_kwargs())
                    report(result, name)
                    return result
                if args.command == "evaluate":
                    score(args.representation)
                else:
                    compare(score(args.a), score(args.b), args.a, args.b, config.min_effect)
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
