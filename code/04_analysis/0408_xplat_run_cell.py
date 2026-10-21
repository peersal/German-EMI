"""Runner (cell 21 of 0403_analysis.ipynb) that draws the cross-platform figure on the full data and writes results/cross_platform_split_afd.csv.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
# Cross-platform EMI figure: per-platform EMI trend + before/after-AfD
# boxplots (A = Twitter 2015-2025, B = Parliament 2000-2025) with
# frequency panels. The parliament series is the paper's model re-run on
# all Bundestag speeches 2000-2025 (data/revision/bundestag_2000_2025_
# scored.csv; z-standardised over 2000-2025, quarterly r = 0.997 with
# the paper's 2015-2025 series). The split statistics (quarterly means
# and documents, before vs after 24 Oct 2017) go to
# results/cross_platform_split_afd.csv.
#
# SAMPLE_N: set to an int (e.g. 50_000) for a fast TEST run on a cached
# random Twitter subsample (data/_samples/, created once); set to None
# for the full data / final paper figure. Sample runs save to
# *_sample.{pdf,png} so the paper figure is never overwritten.
import os

SAMPLE_N = None  # None -> full data (paper figure)


def _load_xplat_sample(path, usecols, sample_n, seed=42):
    if sample_n is None:
        return pd.read_csv(path, usecols=usecols)
    cache = os.path.join(
        "data", "_samples",
        os.path.basename(path).replace(".csv", f"_sample{sample_n}.csv"))
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if set(usecols).issubset(df.columns):
            return df[list(usecols)]
    df = pd.read_csv(path, usecols=usecols)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    return df


df_twitter_emi = _load_xplat_sample(
    "data/twitter_nhb.csv",
    ["created_at", "emi_w2vparliament", "leaning"], SAMPLE_N)
df_parliament_emi = (
    pd.read_csv("data/revision/bundestag_2000_2025_scored.csv",
                usecols=["date", "emi", "leaning"])
    .rename(columns={"emi": "emi_w2vparliament"}))

_suffix = "" if SAMPLE_N is None else "_sample"
fig = plot_emi_cross_platform(
    df_twitter_emi, df_parliament_emi,
    split_date="2017-10-24",
    save_path=f"results/emi_twitter_parliament{_suffix}.pdf",
    stats_path=f"results/cross_platform_split_afd{_suffix}.csv",
)
fig.savefig(f"results/emi_twitter_parliament{_suffix}.png", dpi=300,
            bbox_inches="tight", facecolor="white")
