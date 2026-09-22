"""Date-only shortcut and size-matched leave-one-interval-out diagnostics.

LOIO is a temporal composition diagnostic, not spatially blocked deployment
evaluation. Seen/unseen arms share test rows and training size; only the seen
arm includes held-back rows from the tested interval. Endpoint overlap is
derived from the manifest, never hardcoded for a particular site.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from heritage_watch.config import load_config
from heritage_watch.features import assemble, load_features
from heritage_watch.protocol import (_limit_threads, default_classifier, evaluate,
                                     report, spatial_blocks)


def date_only(config, meta):
    # Encoding is fixed one-hot vocabulary, not a learned statistic or scaler.
    pairs = sorted(set(meta["pair"]))
    X = np.eye(len(pairs))[np.searchsorted(pairs, meta["pair"])]
    result = evaluate(X, meta["y"], meta, **config.evaluation_kwargs())
    report(result, "Date only (no imagery), official jittered protocol")
    labels, counts = np.unique(meta["y"], return_counts=True)
    floor = f1_score(meta["y"], np.full(len(X), labels[counts.argmax()]), average="macro")
    print(f"majority-class floor: {floor:.4f}")
    return dict(n=len(X), classes=config.classes, macro_f1=result.macro_f1,
                per_replicate=result.per_replicate, majority_floor=float(floor))


def loio(config, X, meta, manifest, cells, reps=6, n_train=500):
    if reps < 2 or n_train < 2:
        raise ValueError("LOIO needs at least two replicates and two training rows")
    pair, y = meta["pair"], meta["y"]
    cells = np.asarray(cells)
    records = {r["fid"]: r for r in manifest["records"]}
    endpoints = {}
    for fid, p in zip(meta["fid"], pair):
        if int(fid) not in records:
            raise ValueError(f"manifest missing feature fid {fid}")
        r = records[int(fid)]
        if r["pair"] != p:
            raise ValueError("manifest and feature pair metadata differ")
        endpoints.setdefault(p, set()).update([r["t1"], r["t2"]])
    results = {}
    with _limit_threads():
        for p in sorted(set(pair)):
            idx_p = np.flatnonzero(pair == p)
            if len(idx_p) < 2:
                raise ValueError(f"interval {p}: too few rows to split")
            seen, unseen, sizes = [], [], []
            for r in range(reps):
                rng = np.random.default_rng(7000 + 13 * r)
                # Test rows are WHOLE spatial cells, and those cells are removed
                # from both arms' training pools. Splitting an interval's rows at
                # random instead would put training rows metres from test rows,
                # so the contrast would measure proximity, not date novelty.
                order = rng.permutation(np.unique(cells[idx_p]))
                take, want = [], len(idx_p) // 2
                for c in order:
                    if sum(int((cells[idx_p] == t).sum()) for t in take) >= want:
                        break
                    take.append(c)
                te_cells = set(int(c) for c in take)
                in_te = np.isin(cells[idx_p], list(te_cells))
                test, rest = idx_p[in_te], idx_p[~in_te]
                if len(test) == 0 or len(rest) == 0:
                    continue
                pool = np.flatnonzero((pair != p) & ~np.isin(cells, list(te_cells)))
                n_tr = min(n_train, len(pool))
                if n_tr < 2:
                    raise ValueError(f"interval {p}: training pool exhausted by test cells")
                un = rng.choice(pool, size=n_tr, replace=False)
                k = min(len(rest), n_tr)
                se = np.concatenate([rest[:k], rng.choice(pool, size=n_tr - k, replace=False)])
                for train, scores in ((un, unseen), (se, seen)):
                    if len(set(y[train])) < 2:
                        raise ValueError(f"interval {p}: training arm contains fewer than two classes")
                    clf = default_classifier().fit(X[train], y[train])
                    scores.append(f1_score(y[test], clf.predict(X[test]), average="macro",
                                           labels=np.unique(y[test]), zero_division=0))
                sizes.append((int(n_tr), int(len(test))))
            if not seen:
                raise ValueError(f"interval {p}: no usable spatially blocked split")
            delta = np.asarray(unseen) - seen
            shares = any(endpoints[p] & endpoints[q] for q in endpoints if q != p)
            results[str(p)] = dict(seen=seen, unseen=unseen, delta=float(delta.mean()),
                                   shares_endpoint=shares,
                                   n_test_mean=float(np.mean([s[1] for s in sizes])),
                                   n_train_mean=float(np.mean([s[0] for s in sizes])),
                                   n_train_min=int(min(s[0] for s in sizes)),
                                   reps_used=len(seen))
            print(f"{p}: seen={np.mean(seen):.4f} unseen={np.mean(unseen):.4f} "
                  f"delta={delta.mean():+.4f} shares_endpoint={shares}", flush=True)
    deltas = np.array([v["delta"] for v in results.values()])
    clean = [v["delta"] for v in results.values() if not v["shares_endpoint"]]
    # sample SD: n here is the number of intervals, typically 10
    se2 = 2 * deltas.std(ddof=1) / np.sqrt(len(deltas))
    unanimous = bool(np.all(deltas < 0) or np.all(deltas > 0))
    out = dict(per_interval=results, mean_delta=float(deltas.mean()), two_se=float(se2),
               se_over="interval means",
               resolved=bool(unanimous and abs(deltas.mean()) > se2
                             and abs(deltas.mean()) >= config.min_effect),
               clean_only_delta=float(np.mean(clean)) if clean else None,
               reps=reps, n_train_cap=n_train)
    print(f"mean unseen-seen: {deltas.mean():+.4f}; 2*SE={se2:.4f}; "
          f"resolved={out['resolved']}; no-shared-endpoint intervals={len(clean)}")
    if not clean:
        print("  no interval is endpoint-clean: the resolved scenes chain, so every "
              "'unseen' arm has still seen one endpoint image via a neighbouring pair")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("control", choices=("date-only", "loio"), help="Which diagnostic to run.")
    ap.add_argument("--config", required=True, help="Site configuration; classes must match caches.")
    ap.add_argument("--features", required=True, nargs="+", help="Aligned encoder caches.")
    ap.add_argument("--representation", default="merged", help="LOIO representation (default merged, as in report).")
    ap.add_argument("--manifest", help="LOIO manifest, required to measure actual endpoint overlap.")
    ap.add_argument("--replicates", type=int, default=6, help="LOIO paired replicates per interval (default 6).")
    ap.add_argument("--train-size", type=int, default=500, help="Identical LOIO training size for both arms (default 500).")
    ap.add_argument("--out", default="out/control.json", help="Diagnostic JSON output.")
    args = ap.parse_args()
    try:
        config = load_config(args.config)
        blocks, meta = load_features(args.features, config)
        if args.control == "date-only":
            result = date_only(config, meta)
        else:
            if not args.manifest:
                ap.error("loio requires --manifest")
            cells = spatial_blocks(meta["x"], meta["yy"])
            result = loio(config, assemble(blocks, args.representation), meta,
                          json.loads(Path(args.manifest).read_text()), cells,
                          args.replicates, args.train_size)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    except (ValueError, OSError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
