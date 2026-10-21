"""Cross-platform main figure (cell 12 of 0403_analysis.ipynb): quarterly EMI series for Twitter and the Bundestag with before/after AfD-entry boxplots, election lines and pre-election levels. Run with `python code/04_analysis/0406_run_figure_cells.py fig`.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
# ======================================================================
# Cross-platform EMI figure (Parliament vs Twitter) -- one block per
# platform (A = Twitter, B = Parliament):
#   left, tall:   quarterly mean EMI with bootstrapped 95% CIs, federal
#                 elections (dashed), the AfD's entry into the Bundestag
#                 on 24 Oct 2017 (red) and, Bundestag only, per legislative period, the
#                 pre-election level (mean of the 12 months before each
#                 election, last 90 days excluded) as a dashed horizontal
#                 line running from the previous election to that election
#   left, short:  quarterly number of documents per leaning plus all
#                 documents (same time axis)
#   right, tall:  boxplots of the quarterly mean EMI before vs after the
#                 AfD's entry (split 24 Oct 2017), sharing the y-axis
#   right, short: total number of documents per leaning (horizontal bars)
# Parliament uses the full 2000-2025 series (EMI re-scored with the paper's
# model and standardised over 2000-2025; quarterly r = 0.997 with the
# paper's 2015-2025 series); Twitter covers 2015-2025.
# EMI is z-standardised WITHIN each corpus, so absolute levels are not
# directly comparable across platforms; trends/changes are.
# ======================================================================
from matplotlib.colors import to_rgba
from matplotlib.ticker import (FuncFormatter, LogLocator,
                               MaxNLocator, NullFormatter)

_XPLAT_STYLE = {
    "Twitter":    {"color": "#AA4499", "marker": "o"},
    "Parliament": {"color": "#117733", "marker": "s"},
}
_XPLAT_ELECTIONS = {2002: "2002-09-22", 2005: "2005-09-18", 2009: "2009-09-27",
                    2013: "2013-09-22", 2017: "2017-09-24", 2021: "2021-09-26",
                    2025: "2025-02-23"}
_XPLAT_AFD_ENTRY = pd.Timestamp("2017-10-24")
_XPLAT_XLIM = {"Twitter":    (pd.Timestamp("2015-01-01"), pd.Timestamp("2026-01-01")),
               "Parliament": (pd.Timestamp("2000-01-01"), pd.Timestamp("2026-01-01"))}


def _xplat_prep(df, date_col, emi_col, extra_cols=()):
    cols = [date_col, emi_col, *[c for c in extra_cols if c in df.columns]]
    d = df[cols].copy()
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    d[emi_col] = pd.to_numeric(d[emi_col], errors="coerce")
    return d.dropna(subset=["_date", emi_col]).sort_values("_date")


def _xplat_boot_means(vals, n_boot=1000, seed=0):
    """Bootstrap distribution of the mean (vectorised, chunked)."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, int(5e6 // n))
    i = 0
    while i < n_boot:
        k = min(chunk, n_boot - i)
        idx = rng.integers(0, n, size=(k, n))
        means[i:i + k] = vals[idx].mean(axis=1)
        i += k
    return means


def _xplat_ci(boot, ci=95):
    return (np.percentile(boot, (100 - ci) / 2),
            np.percentile(boot, 100 - (100 - ci) / 2))


def _xplat_cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2
                  + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan


def _xplat_fmt_count(x, _pos=None):
    if x >= 1e6:
        s = f"{x / 1e6:.1f}M"
        return s.replace(".0M", "M")
    if x >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def _xplat_pre_election_levels(d, emi_col, xlim):
    """Mean EMI in the 12 months before each election (last 90 days excluded),
    drawn from the previous election (or the window start) to the election."""
    out = []
    prev = xlim[0]
    for year, day in _XPLAT_ELECTIONS.items():
        e = pd.Timestamp(day)
        if e <= xlim[0] or e > xlim[1]:
            continue
        win = d[(d["_date"] >= e - pd.Timedelta(days=365))
                & (d["_date"] < e - pd.Timedelta(days=90))][emi_col]
        if len(win) >= 50:
            out.append((year, prev, e, float(win.mean()), int(len(win))))
        prev = e
    return out


def plot_emi_cross_platform(df_twitter, df_parliament,
                            emi_col="emi_w2vparliament",
                            date_col_twitter="created_at",
                            date_col_parliament="date",
                            leaning_col="leaning",
                            split_date=_XPLAT_AFD_ENTRY,
                            n_boot=1000,
                            save_path=None,
                            stats_path=None):
    """Cross-platform EMI comparison, one block per platform (see header).
    Returns the figure; the before/after-split statistics of the quarterly
    means and of the documents are printed and (optionally) saved."""
    _white_style = {
        "figure.facecolor":  "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "text.color":     "black",
        "axes.edgecolor":    "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    }
    split_date = pd.Timestamp(split_date)
    split_q = pd.Timestamp(year=split_date.year, month=3 * ((split_date.month - 1) // 3) + 1, day=1)
    with plt.rc_context(_white_style):
        datasets = {
            "Twitter":    _xplat_prep(df_twitter, date_col_twitter, emi_col,
                                      extra_cols=(leaning_col,)),
            "Parliament": _xplat_prep(df_parliament, date_col_parliament,
                                      emi_col, extra_cols=(leaning_col,)),
        }
        leaning_order = ["left", "middle", "right"]
        stats_rows = []

        fig = plt.figure(figsize=(26, 14))
        gs = GridSpec(5, 2, height_ratios=[3, 1, 0.1, 3, 1],
                      width_ratios=[3.1, 1], hspace=0.3, wspace=0.06)
        block_rows = {"Twitter": (0, 1), "Parliament": (3, 4)}
        for letter, (platform, d) in zip("AB", datasets.items()):
            style = _XPLAT_STYLE[platform]
            xlim = _XPLAT_XLIM[platform]
            d = d[(d["_date"] >= xlim[0]) & (d["_date"] < xlim[1])]
            r_emi, r_frq = block_rows[platform]
            ax_e = fig.add_subplot(gs[r_emi, 0])
            ax_b = fig.add_subplot(gs[r_emi, 1], sharey=ax_e)
            ax_f = fig.add_subplot(gs[r_frq, 0], sharex=ax_e)
            ax_t = fig.add_subplot(gs[r_frq, 1])

            # ---- EMI time series: quarterly means + bootstrap CI ---------
            p_i = 0 if platform == "Twitter" else 1
            grouped = d.set_index("_date")[emi_col].resample("QS")
            qs, qstarts, means, m_lo, m_hi = [], [], [], [], []
            for q_i, (qstart, series) in enumerate(grouped):
                series = series.dropna()
                boot = _xplat_boot_means(series.to_numpy(), n_boot=n_boot,
                                         seed=p_i * 10000 + q_i)
                if boot is None:
                    continue
                qstarts.append(qstart)
                qs.append(qstart + pd.Timedelta(days=45))  # quarter midpoint
                means.append(series.mean())
                lo, hi = _xplat_ci(boot)
                m_lo.append(lo); m_hi.append(hi)
            ax_e.plot(qs, means, color="black", marker=style["marker"],
                      markersize=6 if platform == "Parliament" else 7,
                      linewidth=2.3, zorder=3, label=platform)
            ax_e.fill_between(qs, m_lo, m_hi, color="black",
                              alpha=0.18, zorder=2)
            # elections (dashed), AfD entry (solid), pre-election levels
            for year, day in _XPLAT_ELECTIONS.items():
                e = pd.Timestamp(day)
                if xlim[0] < e < xlim[1] and year != 2017:
                    for ax in (ax_e, ax_f):
                        ax.axvline(e, color="black", lw=0.8, alpha=0.55, ls="--")
            for ax in (ax_e, ax_f):
                ax.axvline(split_date, color="#D62728", lw=1.8, alpha=0.95, ls="-", zorder=4)
            levels = _xplat_pre_election_levels(d, emi_col, xlim)
            if platform == "Parliament":   # pre-election levels only for the Bundestag
                for year, x0, x1, lev, n in levels:
                    ax_e.hlines(lev, x0, x1, color="black", lw=1.4, ls=(0, (4, 3)),
                                alpha=0.75, zorder=4)
            ax_e.axhline(0, color="black", lw=0.8, alpha=0.6, zorder=1)
            ax_e.grid(True, alpha=0.25)
            ax_e.tick_params(labelsize=PLOT_TICK_SIZE)
            ax_e.set_ylabel("EMI score", fontsize=PLOT_LABEL_SIZE,
                            fontweight="bold")
            ax_e.text(0.012, 0.98, letter, transform=ax_e.transAxes,
                      fontsize=PLOT_PANEL_SIZE, fontweight="bold", va="top",
                      ha="left", zorder=10)
            handles = [Line2D([0], [0], color="black", marker=style["marker"],
                              markersize=7, linewidth=2.3, label=platform)]
            if letter == "A":   # AfD-entry legend entry once, in the top panel
                handles.append(Line2D([0], [0], color="#D62728", lw=1.8, label="AfD enters Bundestag"))
            ax_e.legend(handles=handles, frameon=False, loc="upper right",
                        fontsize=PLOT_LEGEND_SIZE + 2, ncol=3,
                        columnspacing=1.2, handlelength=2.2)

            # ---- boxplots of the quarterly means before / after split ---
            qstarts = pd.Series(qstarts)
            qmeans = np.asarray(means)
            before = qmeans[(qstarts < split_q).to_numpy()]
            after = qmeans[(qstarts >= split_q).to_numpy()]
            box_data = [before, after]
            box_labels = ["before AfD entry", "after AfD entry"]
            art = ax_b.boxplot(
                box_data, positions=range(2), widths=0.55,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2.0),
                whiskerprops=dict(color="black", linewidth=1.5),
                capprops=dict(color="black", linewidth=1.5),
                boxprops=dict(edgecolor="black", linewidth=1.5),
                flierprops=dict(marker="o", markersize=5,
                                markerfacecolor="black",
                                markeredgecolor="none", alpha=0.8))
            for patch, alpha in zip(art["boxes"], (0.30, 0.10)):
                patch.set_facecolor(to_rgba("black", alpha))
            ax_b.set_xlim(-0.6, 1.6)
            ax_b.set_xticks(range(2))
            ax_b.set_xticklabels(box_labels, fontsize=PLOT_TICK_SIZE)
            ax_b.tick_params(labelsize=PLOT_TICK_SIZE)
            plt.setp(ax_b.get_yticklabels(), visible=False)
            ax_b.grid(True, axis="y", alpha=0.25)
            y0, y1 = ax_e.get_ylim()
            ax_e.set_ylim(y0, y1 + 0.18 * (y1 - y0))

            # ---- split statistics: quarterly means and documents --------
            docs_before = d[d["_date"] < split_date][emi_col].to_numpy()
            docs_after = d[d["_date"] >= split_date][emi_col].to_numpy()
            tq = stats.ttest_ind(before, after, equal_var=False)
            td = stats.ttest_ind(docs_before, docs_after, equal_var=False)
            mw = stats.mannwhitneyu(docs_before, docs_after)
            stats_rows.append({
                "platform": platform, "split": str(split_date.date()),
                "window_start": str(xlim[0].date()),
                "quarters_before": len(before), "quarters_after": len(after),
                "qmean_before": before.mean(), "qmean_after": after.mean(),
                "qmean_diff": after.mean() - before.mean(),
                "qmedian_before": np.median(before), "qmedian_after": np.median(after),
                "q_welch_t": tq.statistic, "q_welch_p": tq.pvalue,
                "q_cohens_d": _xplat_cohens_d(before, after),
                "docs_before": len(docs_before), "docs_after": len(docs_after),
                "doc_mean_before": docs_before.mean(), "doc_mean_after": docs_after.mean(),
                "doc_diff": docs_after.mean() - docs_before.mean(),
                "doc_welch_t": td.statistic, "doc_welch_p": td.pvalue,
                "doc_cohens_d": _xplat_cohens_d(docs_before, docs_after),
                "doc_mannwhitney_p": mw.pvalue,
                "pre_election_levels": "; ".join(f"{y}: {lev:+.3f} (n={n})" for y, _, _, lev, n in levels),
            })

            # ---- frequency time series per leaning + all ----------------
            cnt_all = d.set_index("_date")[emi_col].resample("QS").size()
            cnt_all = cnt_all[cnt_all > 0]
            ax_f.plot(cnt_all.index + pd.Timedelta(days=45), cnt_all.values,
                      color="black", linewidth=1.8, alpha=0.9, label="All")
            for leaning in leaning_order:
                cnt = (d[d[leaning_col] == leaning]
                       .set_index("_date")[emi_col].resample("QS").size())
                cnt = cnt.reindex(cnt_all.index).where(lambda c: c > 0)  # gaps stay gaps
                ax_f.plot(cnt.index + pd.Timedelta(days=45), cnt.values,
                          color=COLORS.get(leaning, "gray"),
                          marker=MARKERS.get(leaning, "o"), markersize=4,
                          linewidth=1.5, alpha=0.85)
            ax_f.grid(True, alpha=0.25)
            ax_f.set_ylabel("Freq.", fontsize=PLOT_LABEL_SIZE,
                            fontweight="bold")
            ax_f.set_yscale("log")
            ax_f.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax_f.yaxis.set_major_formatter(FuncFormatter(_xplat_fmt_count))
            ax_f.yaxis.set_minor_formatter(NullFormatter())
            ax_f.tick_params(labelsize=PLOT_TICK_SIZE)

            # ---- total documents per leaning (horizontal bars) -----------
            totals = [int((d[leaning_col] == leaning).sum())
                      for leaning in leaning_order]
            ypos = np.arange(len(leaning_order))[::-1]
            ax_t.barh(ypos, totals, height=0.62,
                      color=[COLORS.get(l, "gray") for l in leaning_order],
                      alpha=0.85)
            for y, tot, leaning in zip(ypos, totals, leaning_order):
                ax_t.text(tot + 0.03 * max(totals), y,
                          LEANING_LABELS.get(leaning, leaning.capitalize()),
                          va="center", ha="left", fontsize=PLOT_TICK_SIZE)
            ax_t.text(0.98, 0.95, f"All: {_xplat_fmt_count(len(d))}",
                      transform=ax_t.transAxes, ha="right", va="top",
                      fontsize=PLOT_TICK_SIZE)
            ax_t.set_xlim(0, max(totals) * 1.5)
            ax_t.set_ylim(-0.65, len(leaning_order) - 0.35)
            ax_t.set_yticks([])
            ax_t.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_t.xaxis.set_major_formatter(FuncFormatter(_xplat_fmt_count))
            ax_t.tick_params(labelsize=PLOT_TICK_SIZE)
            ax_t.grid(True, axis="x", alpha=0.25)

            # ---- time axis per platform ---------------------------------
            ax_e.set_xlim(*xlim)
            step = 1 if platform == "Twitter" else 2
            year_ticks = pd.date_range(xlim[0], xlim[1], freq=f"{step}YS")
            ax_f.set_xticks(year_ticks)
            ax_f.set_xticklabels([str(t.year) for t in year_ticks],
                                 fontsize=PLOT_TICK_SIZE)
            plt.setp(ax_e.get_xticklabels(), visible=False)
            ax_f.set_xlabel("Date", fontsize=PLOT_LABEL_SIZE, fontweight="bold")

        stats_df = pd.DataFrame(stats_rows)
        show_cols = ["platform", "quarters_before", "quarters_after", "qmean_before",
                     "qmean_after", "qmean_diff", "q_welch_p", "q_cohens_d",
                     "docs_before", "docs_after", "doc_mean_before", "doc_mean_after",
                     "doc_cohens_d", "doc_welch_p"]
        print(stats_df[show_cols].round(4).to_string(index=False))
        for _, r in stats_df.iterrows():
            print(f"{r.platform} pre-election levels: {r.pre_election_levels}")
        if stats_path:
            stats_df.to_csv(stats_path, index=False)
            print(f"Split statistics saved to {stats_path}")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight",
                        facecolor="white")
            print(f"Saved to {save_path}")
        plt.show()
        return fig
