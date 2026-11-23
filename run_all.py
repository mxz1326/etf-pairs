"""Run the whole study and write tables and figures to results/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etfpairs.study import BASE_COST_BP, TEST, TRAIN, run_all
from etfpairs.data import flag_suspect_days, load_study_data

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


# Format a DataFrame as a markdown table
def fmt_table(df, floatfmt=2):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([head, sep, *body])


# Fig 1: the spread over time; Fig 2: cumulative P&L, realistic vs same-close fill
def figures(o):
    s = o["series"]
    spread_bp = (s["spread"] - s["spread"].iloc[0]) * 1e4
    daily_bp = s["spread_ret"] * 1e4

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(spread_bp.index, spread_bp.values, lw=0.8, color="#1f4e79")
    ax[0].set_ylabel("IOZ minus STW\ncumulative, bp")
    ax[0].set_title("Total-return spread between two ASX 200 ETFs")
    ax[1].plot(daily_bp.index, daily_bp.values, lw=0.5, color="#7f7f7f")
    ax[1].set_ylabel("daily difference, bp")
    ax[1].set_ylim(-150, 150)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_spread.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for res, label, col in ((s["res_lag0"], "same-close fill (not achievable)", "#c55a11"),
                            (s["res"], "next-close fill (realistic)", "#1f4e79")):
        ax.plot(res.index, res["net"].cumsum() * 1e4, label=label, color=col, lw=1)
    ax.axvline(pd.Timestamp(TEST[0]), color="k", ls="--", lw=0.8)
    ax.text(pd.Timestamp(TEST[0]), ax.get_ylim()[1] * 0.9, "  test period", fontsize=8)
    ax.set_ylabel("cumulative net P&L, bp of A$1 per leg")
    ax.set_title(f"Chosen rule after {BASE_COST_BP:.0f} bp per leg costs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_pnl.png", dpi=150)
    plt.close(fig)


# Run the study, save CSVs, figures and summary.md, and print the summary
def main():
    o = run_all()
    d = o["describe"]
    o["grid"].to_csv(OUT / "grid_train.csv", index=False)
    o["by_cost"].to_csv(OUT / "chosen_rule_by_cost.csv", index=False)
    o["variants"].to_csv(OUT / "test_variants.csv", index=False)
    _, _, _, ret = load_study_data()
    flag_suspect_days(ret).round(1).to_csv(OUT / "days_over_50bp.csv")
    figures(o)

    lines = ["# Results", "",
             f"Sample {d['first_day']} to {d['last_day']} ({d['trading_days']} trading days). "
             f"Parameters chosen on {TRAIN[0]} to {TRAIN[1]}, tested once from {TEST[0]}.", "",
             "## The spread", ""]
    lines += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in d.items()]
    lines += ["", "## Training grid (net of 2 bp per leg, next-close fill)", "",
              fmt_table(o["grid"][["window", "entry", "ann_ret_bp", "sharpe", "t_stat", "trades",
                                   "days_in_market_pct", "breakeven_cost_bp"]]), "",
              f"Chosen: window {o['chosen']['window']} days, entry |z| > {o['chosen']['entry']}.", "",
              "## Chosen rule by cost level", "",
              fmt_table(o["by_cost"][["period", "cost_bp_per_leg", "ann_ret_bp", "ann_vol_bp", "sharpe",
                                      "t_stat", "trades", "gross_bp_per_trade"]]), "",
              "## Test period: fill assumption and dividend blackout", "",
              fmt_table(o["variants"]), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()