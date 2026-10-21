"""Before/after AfD-entry comparison of the Bundestag series on the full 2000-2025 corpus (cell 22 of 0403_analysis.ipynb); writes results/parliament/tables/temporal_comparison_2000_*.csv.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
# Before/after comparison of the Bundestag series on the FULL 2000-2025 corpus
# (the paper's scoring model re-run on all plenary speeches since 2000,
# z-standardised over 2000-2025; data/revision/bundestag_2000_2025_scored.csv).
# Split = the AfD's entry into the Bundestag (parliament_config["split_date"]).
# Tables: results/parliament/tables/temporal_comparison_2000_{overall,by_leaning}.csv
df_parl_2000 = load_and_prepare_data(
    "data/revision/bundestag_2000_2025_scored.csv",
    date_column="date", group_column="party", leaning_column="leaning",
    id_column="politicianId", dropna_cols=["party", "leaning", "emi"])
res_2000 = temporal_comparison(
    df_parl_2000, emi_col="emi", split_date=parliament_config["split_date"],
    methods=["ttest", "cohens_d", "bootstrap", "mannwhitney"])
pd.DataFrame([res_2000["overall"]]).to_csv(
    "results/parliament/tables/temporal_comparison_2000_overall.csv", index=False)
pd.DataFrame(res_2000["by_leaning"]).T.to_csv(
    "results/parliament/tables/temporal_comparison_2000_by_leaning.csv")
