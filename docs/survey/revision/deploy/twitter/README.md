# EMI validation survey: twitter

Potato project ({potato-annotation 2.8.2}) for rating Kurznachrichten (Tweets) von Politikerinnen und Politikern (2015–2025) on the two EMI items of the original survey.

| Parameter | Value |
|---|---|
| Instances | 300 blinded texts in `data_files/survey_sample.csv` (id, text); party/EMI/date in `data_files/survey_sample_key.csv`, never shown |
| Sampling | 50 per party (CDU/CSU, SPD, Greens, FDP, Left, AfD), spread evenly over the global EMI quintiles; see `../sample_summary.md` |
| Items | Evidenz basiert, Intuition basiert (multirate grid, option order randomised) |
| Scale | Stimme überhaupt nicht zu / Stimme nicht zu / Neutral / Stimme zu / Stimme voll und ganz zu → code as −2 … +2 (or 0–4) |
| Annotations per item | 5 |
| Items per annotator | 20, random assignment |
| Participants needed | 300 × 5 / 20 = **75** completed |
| Attention checks | 4 gold items in `surveyflow/gold_standards.jsonl` (the two instruction examples + one evidence-based and one intuitive text), mixed in ~every 10 items |
| Flow | intro → instructions → consent → annotation → end (German, texts from the original EMI survey) |

## Run locally

```bash
cd docs/survey/revision/twitter
potato start config.yaml -p 8002
```

(requires `pip install potato-annotation==2.8.2`.)

Then open <http://localhost:8002/?PROLIFIC_PID=test1> (any PID value = one annotator; a new value simulates a new participant).

Reset between test runs: stop the server and delete `annotation_output/`, `project.sqlite*`, `layouts/`.

## Output

- `annotation_output/pilot/exports/csv/annotations.csv`: one row per user × instance with `Rating.Evidenz basiert` and `Rating.Intuition basiert`.
- Rows whose `instance_id` starts with `gold_` are the attention checks; drop them from the analysis and use them to screen annotators (the admin dashboard tracks gold accuracy per user; the check is an exact label match, so treat it as a screening signal, not a hard filter).
- Join the analysis to `data_files/survey_sample_key.csv` on `id` for party, leaning, date and embedding EMI.

## Before fielding on Prolific

1. Set `completion_code:` in `config.yaml` to the code of the Prolific study.
2. Prolific external study URL: `https://<server>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`. Screeners: German (fluent), resident in Germany, 18+.
3. Optional Prolific API integration: copy `configs/prolific_config_template.yaml` to `configs/prolific_config.yaml` (git-ignored), fill in token and study id, switch `login.type` to `prolific` and uncomment the `prolific:` block.
4. Deploy behind HTTPS (reverse proxy or `--ssl-cert/--ssl-key`).
