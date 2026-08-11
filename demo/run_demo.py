"""Demo for the German-EMI analysis pipeline.

Part 1 (Bundestag): the constrained breakpoint linear mixed-effects
regression of the paper (Methods, Equation 2) on a small real subset of
the contemporary Bundestag corpus (demo_data.csv: 2,932 speeches by 40
politicians with precomputed EMI scores).

Part 2 (Twitter): the baseline linear mixed-effects model (Methods,
Equation 1, without the engagement covariate, which is not included in
the shared file) on demo_data_twitter.csv (6,000 tweets by 40
pseudonymised politician accounts; tweet IDs only, no text).

Usage:
    python run_demo.py

Outputs (written next to this script):
    demo_output/model_summary_parliament.txt
    demo_output/emi_trends_parliament.png
    demo_output/model_summary_twitter.txt
    demo_output/emi_trends_twitter.png
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_output")
os.makedirs(OUT, exist_ok=True)

BREAKPOINTS = {  # ~1 year before each federal election (paper, Methods)
    "elec2017": pd.Timestamp("2017-07-01"),
    "elec2021": pd.Timestamp("2021-07-01"),
    "elec2025": pd.Timestamp("2025-01-01"),
}
COLORS = {"left": "#E66101", "middle": "#4D4D4D", "right": "#0072B2"}
LABELS = {"left": "Left", "middle": "Centre", "right": "Right"}
EMI = "emi_w2vparliament"


def add_quarterly_lag(df, actor_col):
    """Previous-quarter mean EMI per actor (absorbs autocorrelation)."""
    df["time_quarterly"] = df["date"].dt.year * 10 + df["date"].dt.quarter
    q = (df.groupby([actor_col, "time_quarterly"])[EMI].mean()
           .reset_index().rename(columns={EMI: "_qm"}))
    q["emi_lag1"] = q.groupby(actor_col)["_qm"].shift(1)
    return (df.merge(q[[actor_col, "time_quarterly", "emi_lag1"]],
                     on=[actor_col, "time_quarterly"])
              .dropna(subset=["emi_lag1"]).reset_index(drop=True))


def fit_and_report(df, actor_col, formula, tag):
    res = MixedLM.from_formula(formula, groups=df[actor_col],
                               data=df).fit(reml=False)
    null = MixedLM.from_formula(f"{EMI} ~ 1", groups=df[actor_col],
                                data=df).fit(reml=False)
    icc = null.cov_re.iloc[0, 0] / (null.cov_re.iloc[0, 0] + null.scale)
    path = os.path.join(OUT, f"model_summary_{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(res.summary()) + f"\n\nICC (null model): {icc:.3f}\n")
    print(res.summary().tables[1])
    print(f"ICC (null model): {icc:.3f}")
    return res


def plot_trends(df, tag, min_docs):
    fig, ax = plt.subplots(figsize=(10, 5))
    for leaning in ["left", "middle", "right"]:
        d = df[df["leaning"] == leaning]
        g = (d.set_index("date")[EMI].resample("QS").agg(["mean", "count"]))
        g = g[g["count"] >= min_docs]
        ax.plot(g.index, g["mean"], color=COLORS[leaning], marker="o",
                markersize=4, linewidth=1.5, label=LABELS[leaning])
    for ts in BREAKPOINTS.values():
        ax.axvline(ts, color="black", lw=0.8, alpha=0.55, ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("EMI score")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"emi_trends_{tag}.png"), dpi=200)
    plt.close(fig)


t0 = time.time()

# ===== Part 1: Bundestag, constrained breakpoint model (Equation 2) ===
print("=== Bundestag: breakpoint mixed model (Equation 2) ===")
df = pd.read_csv(os.path.join(HERE, "demo_data.csv"))
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df)} speeches, {df['politicianId'].nunique()} "
      f"politicians, {df['date'].min().date()} .. {df['date'].max().date()}")
df = add_quarterly_lag(df, "politicianId")

tm = df["time_quarterly"].mean()
df["time_centered"] = df["time_quarterly"] - tm
terms = ["time_centered", "leaning", "time_centered:leaning"]
for name, ts in BREAKPOINTS.items():
    bp = ts.year * 10 + ts.quarter
    df[f"slope_{name}"] = np.maximum(0.0, df["time_centered"] - (bp - tm))
    df[f"level_{name}"] = (df["time_quarterly"] >= bp).astype(float)
    terms += [f"slope_{name}", f"level_{name}"]
terms.append("emi_lag1")
df["leaning"] = pd.Categorical(df["leaning"],
                               categories=["middle", "left", "right"])
print(f"Fitting MixedLM (N={len(df)}) ...")
fit_and_report(df, "politicianId", f"{EMI} ~ " + " + ".join(terms),
               "parliament")
plot_trends(df, "parliament", min_docs=3)

# ===== Part 2: Twitter, baseline linear model (Equation 1) ============
print("\n=== Twitter: baseline mixed model (Equation 1, no engagement "
      "covariate) ===")
tw = pd.read_csv(os.path.join(HERE, "demo_data_twitter.csv"),
                 dtype={"id": str})
tw["date"] = pd.to_datetime(tw["date"])
print(f"Loaded {len(tw)} tweets, {tw['actor'].nunique()} actors, "
      f"{tw['date'].min().date()} .. {tw['date'].max().date()}")
tw = add_quarterly_lag(tw, "actor")
tw["time_centered"] = tw["time_quarterly"] - tw["time_quarterly"].mean()
tw["leaning"] = pd.Categorical(tw["leaning"],
                               categories=["middle", "left", "right"])
print(f"Fitting MixedLM (N={len(tw)}) ...")
fit_and_report(tw, "actor",
               f"{EMI} ~ time_centered + leaning + time_centered:leaning"
               " + emi_lag1", "twitter")
plot_trends(tw, "twitter", min_docs=5)

print(f"\nOutputs written to {OUT}")
print(f"Total run time: {time.time() - t0:.1f} s")
