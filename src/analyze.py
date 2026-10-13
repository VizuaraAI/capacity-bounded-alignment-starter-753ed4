"""Turn a directory of result files into the paper's tables and figures.

    python -m src.analyze --results-dir results --out-dir analysis

Produces:
  analysis/superficiality.csv   one row per model rung and evaluation axis
  analysis/table2.tex           the LaTeX body of the paper's main results table
  analysis/fig_sindex.png       Superficiality Index against model scale

Run it as soon as the first three result files exist. An early, ugly version of
the main table is far more useful than a late, beautiful one: it tells you
whether the experiment can answer the question at all.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import metrics
from . import model_zoo


def load_results(results_dir: Path) -> dict:
    """Index result files by (model_key, intervention)."""
    index = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("tds__"):
            continue
        payload = json.loads(path.read_text())
        key = (payload.get("model_key"), payload.get("intervention"))
        if None in key:
            print(f"[analyze] skipping {path.name}: missing model_key or intervention")
            continue
        index[key] = payload
    return index


def axis_scores(payload: dict, axis: str) -> list[float]:
    """Concatenate the per example scores of every task on one axis.

    Concatenating keeps the pairing across interventions intact as long as the
    task list and seed match, which the evaluator enforces by construction.
    """
    out = []
    for task in sorted(payload.get(axis, {})):
        out.extend(payload[axis][task]["scores"])
    return out


def compute_rows(index: dict, base_name: str, ceiling_name: str,
                 resamples: int, ci_level: float) -> list[dict]:
    models = sorted({k[0] for k in index}, key=lambda m: (
        model_zoo.get(m).params if m in model_zoo.LADDER else 0
    ))
    rows = []
    for model_key in models:
        for axis in ("format", "capability"):
            triple = [(model_key, base_name), (model_key, "ica"),
                      (model_key, ceiling_name)]
            if not all(t in index for t in triple):
                missing = [t[1] for t in triple if t not in index]
                print(f"[analyze] {model_key}: cannot compute S, missing {missing}")
                continue
            base = axis_scores(index[triple[0]], axis)
            ica = axis_scores(index[triple[1]], axis)
            sft = axis_scores(index[triple[2]], axis)
            n = min(len(base), len(ica), len(sft))
            if n == 0:
                continue
            res = metrics.superficiality_index(
                base[:n], ica[:n], sft[:n], resamples=resamples, ci_level=ci_level
            )
            spec = model_zoo.get(model_key) if model_key in model_zoo.LADDER else None
            rows.append({
                "model": model_key,
                "label": spec.label if spec else model_key,
                "params": spec.params if spec else None,
                "axis": axis,
                **res.as_dict(),
            })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    columns = ["model", "label", "params", "axis", "p_base", "p_ica", "p_sft",
               "value", "ci_low", "ci_high", "n", "status"]
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for col in columns:
            v = row.get(col)
            cells.append("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[analyze] wrote {path}")


def write_latex_table(rows: list[dict], path: Path) -> None:
    """Emit the table body, ready to paste over the placeholder rows in main.tex."""
    by_model = defaultdict(dict)
    for row in rows:
        by_model[(row["params"] or 0, row["label"])][row["axis"]] = row

    def cell(row: dict | None, field: str) -> str:
        if row is None or row.get(field) is None:
            return "XX.X"
        return f"{row[field]:.3f}" if field == "value" else f"{100 * row[field]:.1f}"

    lines = []
    for (_, label) in sorted(by_model):
        fmt = by_model[(_, label)].get("format")
        cap = by_model[(_, label)].get("capability")
        lines.append(
            f"{label} & {cell(fmt, 'p_base')} & {cell(fmt, 'p_ica')} & "
            f"{cell(fmt, 'p_sft')} & {cell(fmt, 'value')} & "
            f"{cell(cap, 'p_base')} & {cell(cap, 'p_ica')} & "
            f"{cell(cap, 'p_sft')} & {cell(cap, 'value')} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[analyze] wrote {path}")


def plot_index(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = {"format": "#4878a8", "capability": "#c4622d"}
    plotted = False
    for axis in ("format", "capability"):
        pts = [r for r in rows
               if r["axis"] == axis and r["value"] is not None and r["params"]]
        if not pts:
            continue
        pts.sort(key=lambda r: r["params"])
        xs = [r["params"] / 1e9 for r in pts]
        ys = [r["value"] for r in pts]
        lo = [max(0.0, r["value"] - (r["ci_low"] or r["value"])) for r in pts]
        hi = [max(0.0, (r["ci_high"] or r["value"]) - r["value"]) for r in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                    color=colors[axis], label=f"{axis} axis")
        plotted = True

    if not plotted:
        print("[analyze] no usable index values yet, skipping plot")
        plt.close(fig)
        return

    ax.axhline(1.0, linestyle="--", color="grey", linewidth=1)
    ax.text(ax.get_xlim()[1], 1.0, " SAH", va="center", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions, log scale)")
    ax.set_ylabel("Superficiality Index $S$")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[analyze] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper's tables from results")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results-dir", default="results", dest="results_dir")
    parser.add_argument("--out-dir", default="analysis", dest="out_dir")
    args = parser.parse_args()

    from .train_sft import load_config
    cfg = load_config(args.config)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"{results_dir} does not exist, run src.evaluate first")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_results(results_dir)
    print(f"[analyze] loaded {len(index)} result files")
    rows = compute_rows(
        index,
        base_name=cfg["analysis"]["reference_base"],
        ceiling_name=cfg["analysis"]["reference_ceiling"],
        resamples=int(cfg["analysis"]["bootstrap_resamples"]),
        ci_level=float(cfg["analysis"]["ci_level"]),
    )
    if not rows:
        raise SystemExit(
            "no complete base/ica/sft triples found. Each rung needs all three "
            "interventions evaluated before S can be computed."
        )

    write_csv(rows, out_dir / "superficiality.csv")
    write_latex_table(rows, out_dir / "table2.tex")
    plot_index(rows, out_dir / "fig_sindex.png")

    cap = [(r["params"], r["value"]) for r in rows
           if r["axis"] == "capability" and r["value"] is not None]
    if len(cap) >= 3:
        elasticity = metrics.capacity_elasticity([p for p, _ in cap], [s for _, s in cap])
        print(f"[analyze] capability-axis S slope: {elasticity['slope']:.3f} per decade "
              f"of parameters (R^2 = {elasticity['r_squared']:.2f})")
        (out_dir / "elasticity.json").write_text(json.dumps(elasticity, indent=2))


if __name__ == "__main__":
    main()
