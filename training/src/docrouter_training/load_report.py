"""Analyse d'une campagne de charge Locust (itération 5) : un résumé par palier d'utilisateurs.

    uv run docrouter-load-report reports/load/requests.csv --label "onnx-int8 2 vCPU"

Pour chaque palier : débit (documents/s), latence p50 / p95 / p99, taux d'erreur, précision.
Les premières secondes de chaque palier sont ignorées (le temps que la charge s'installe).
Sorties : tableau Markdown, graphiques, et un run MLflow (expérience `docrouter-load`).
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EXPERIMENT = "docrouter-load"


@dataclass
class Step:
    users: int
    requests: int
    throughput: float  # réponses réussies par seconde
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    accuracy: float


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            {
                "t": float(r["t"]),
                "users": int(r["users"]),
                "ms": float(r["ms"]),
                "ok": r["ok"] == "1",
                "expected": r.get("expected", ""),
                "predicted": r.get("predicted", ""),
            }
            for r in csv.DictReader(f)
        ]


def summarize(records: list[dict], warmup_s: float = 5.0) -> list[Step]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_step[r["users"]].append(r)
    steps = []
    for users in sorted(u for u in by_step if u > 0):
        rows = by_step[users]
        start = min(r["t"] for r in rows) + warmup_s
        kept = [r for r in rows if r["t"] >= start]
        if len(kept) < 2:
            continue
        duration = max(r["t"] for r in kept) - start
        ok = [r for r in kept if r["ok"]]
        ms = np.array([r["ms"] for r in ok]) if ok else np.array([np.nan])
        steps.append(
            Step(
                users=users,
                requests=len(kept),
                throughput=len(ok) / duration if duration > 0 else 0.0,
                p50_ms=float(np.percentile(ms, 50)),
                p95_ms=float(np.percentile(ms, 95)),
                p99_ms=float(np.percentile(ms, 99)),
                error_rate=1 - len(ok) / len(kept),
                accuracy=(sum(r["predicted"] == r["expected"] for r in ok) / len(ok))
                if ok
                else 0.0,
            )
        )
    return steps


def capacity(steps: list[Step], p95_slo_ms: float) -> dict[str, float]:
    """Débit max, et débit soutenable en respectant l'objectif de latence (p95) sans erreur."""
    best = max(steps, key=lambda s: s.throughput)
    within = [s for s in steps if s.p95_ms <= p95_slo_ms and s.error_rate == 0]
    sustainable = max(within, key=lambda s: s.throughput) if within else None
    return {
        "max_throughput": best.throughput,
        "users_at_max": float(best.users),
        "sustainable_throughput": sustainable.throughput if sustainable else 0.0,
        "users_at_sustainable": float(sustainable.users) if sustainable else 0.0,
        "p95_slo_ms": p95_slo_ms,
        "total_errors": float(sum(round(s.error_rate * s.requests) for s in steps)),
    }


def table(steps: list[Step]) -> str:
    lines = [
        "| Utilisateurs | Requêtes | Débit (doc/s) | p50 (ms) | p95 (ms) | p99 (ms) "
        "| Erreurs | Précision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.users} | {s.requests} | {s.throughput:.1f} | {s.p50_ms:.0f} | {s.p95_ms:.0f} "
        f"| {s.p99_ms:.0f} | {s.error_rate:.1%} | {s.accuracy:.1%} |"
        for s in steps
    ]
    return "\n".join(lines)


def plot(steps: list[Step], out: Path, label: str, p95_slo_ms: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    users = [s.users for s in steps]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(users, [s.throughput for s in steps], marker="o")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(users, [str(u) for u in users])
    ax1.set_xlabel("Utilisateurs simultanés")
    ax1.set_ylabel("Documents traités / s")
    ax1.set_title("Débit")
    ax1.set_ylim(bottom=0)
    ax2.plot(users, [s.p50_ms for s in steps], marker="o", label="p50")
    ax2.plot(users, [s.p95_ms for s in steps], marker="o", label="p95")
    ax2.axhline(p95_slo_ms, color="grey", linestyle="--", label=f"objectif p95 {p95_slo_ms:.0f} ms")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(users, [str(u) for u in users])
    ax2.set_xlabel("Utilisateurs simultanés")
    ax2.set_ylabel("Temps de réponse (ms)")
    ax2.set_title("Temps de réponse")
    ax2.legend()
    fig.suptitle(f"Test de charge — {label}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_compare(runs: dict[str, list[Step]], out: Path, p95_slo_ms: float) -> None:
    """Plusieurs campagnes sur le même graphique (débit et p95 selon la charge)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, steps in runs.items():
        users = [s.users for s in steps]
        ax1.plot(users, [s.throughput for s in steps], marker="o", label=label)
        ax2.plot(users, [s.p95_ms for s in steps], marker="o", label=label)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ticks = sorted({s.users for steps in runs.values() for s in steps})
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlabel("Utilisateurs simultanés")
    ax1.set_ylabel("Documents traités / s")
    ax1.set_title("Débit")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8)
    ax2.axhline(p95_slo_ms, color="grey", linestyle="--")
    ax2.set_yscale("log")
    ax2.set_ylabel("Temps de réponse p95 (ms, échelle log)")
    ax2.set_title(f"Latence p95 (pointillés : objectif {p95_slo_ms:.0f} ms)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "requests", type=Path, nargs="?", help="CSV écrit par load_test/locustfile.py"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="CSV=LIBELLÉ",
        help="trace plusieurs campagnes sur un même graphique (avec --out)",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/images/load_comparison.png"))
    parser.add_argument("--label", default="campagne", help="ex. « onnx-int8 2 vCPU »")
    parser.add_argument("--cpus", type=float, default=None, help="vCPU alloués à l'API")
    parser.add_argument("--p95-slo-ms", type=float, default=500.0)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    if args.compare:
        runs = {}
        for item in args.compare:
            path, _, label = item.partition("=")
            runs[label or Path(path).stem] = summarize(read_records(Path(path)), args.warmup_s)
        plot_compare(runs, args.out, args.p95_slo_ms)
        print(f"Graphique comparatif : {args.out}")
        return
    if args.requests is None:
        parser.error("indiquer un CSV de campagne, ou --compare")
    steps = summarize(read_records(args.requests), args.warmup_s)
    if not steps:
        raise SystemExit("Aucune donnée exploitable.")
    cap = capacity(steps, args.p95_slo_ms)
    if args.cpus:
        cap["sustainable_throughput_per_vcpu"] = cap["sustainable_throughput"] / args.cpus
    out_dir = args.requests.parent / args.requests.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    md = table(steps)
    (out_dir / "summary.md").write_text(
        f"## {args.label}\n\n{md}\n\n```json\n{json.dumps(cap, indent=2)}\n```\n", encoding="utf-8"
    )
    (out_dir / "steps.json").write_text(
        json.dumps([asdict(s) for s in steps], indent=2), encoding="utf-8"
    )
    plot(steps, out_dir / "load_curves.png", args.label, args.p95_slo_ms)
    print(md)
    print(json.dumps(cap, indent=2))

    if not args.no_mlflow:
        import mlflow

        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=args.label):
            mlflow.log_params(
                {"label": args.label, "cpus": args.cpus, "p95_slo_ms": args.p95_slo_ms}
            )
            mlflow.log_metrics(cap)
            for s in steps:
                mlflow.log_metrics(
                    {"throughput": s.throughput, "p50_ms": s.p50_ms, "p95_ms": s.p95_ms},
                    step=s.users,
                )
            mlflow.log_artifacts(str(out_dir))
        print("Run MLflow enregistré : http://localhost:5000")


if __name__ == "__main__":
    main()
