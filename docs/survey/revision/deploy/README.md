# Deployment copies for the VM

Three self-contained potato projects, ready to copy to the VM, all on port 8080 (run one at a time) with
Prolific login via the URL parameters (`login: type: prolific, url_argument: PROLIFIC_PID`) and the Prolific
API block enabled (`prolific: config_file_path: configs/prolific_config.yaml`):

- `parliament_pilot/`: 10 Bundestag chunks, every participant rates all 10, 5 participants needed. Run first.
- `parliament/`: all 300 Bundestag chunks (the 10 pilot chunks are included again, ids p001..p300), 20 per
  participant, 5 ratings per item, 75 participants.
- `twitter/`: 300 tweets, 20 per participant, 5 ratings per item, 75 participants.

## Before copying: paste the Prolific credentials

For each survey, open `configs/prolific_config.yaml` and replace the two placeholders:

```yaml
token: "PASTE_PROLIFIC_API_TOKEN_HERE"      # Prolific workspace -> Settings -> API tokens
study_id: "PASTE_PROLIFIC_STUDY_ID_HERE"    # from the study URL on Prolific (24 hex characters)
```

Then open `config.yaml` and replace

```yaml
completion_code: "PASTE_PROLIFIC_COMPLETION_CODE"   # the completion code shown in the Prolific study
```

`configs/prolific_config.yaml` is git-ignored; do not commit it or paste the token anywhere else.
The three surveys are separate Prolific studies, so each folder gets its own study id and completion code.

## Prolific study settings

- Study URL (external): `https://<vm-host>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`
- Record Prolific IDs via URL parameters.
- Completion: redirect to the Prolific completion URL with the code set above (potato shows the code and
  redirects automatically after 5 s).
- Screeners: German (fluent), resident in Germany, 18+. Places: pilot 5 (6 with margin), parliament 75 (80),
  twitter 75 (83); each participant sees 20 items plus about 3 gold items (pilot: all 10 plus 1 gold item).

## Run on the VM

```bash
pip install potato-annotation==2.8.2
cd parliament_pilot      # then parliament, then twitter
potato start config.yaml -p 8080
```

Run potato behind HTTPS (reverse proxy or `--ssl-cert/--ssl-key`); Prolific requires https.
Before the first real participant: delete any `annotation_output/`, `project.sqlite*` and `layouts/` left
from testing, so assignments start from zero.

## Output

`annotation_output/pilot/exports/csv/annotations.csv` (one row per participant x item; `gold_` rows are the
attention checks). Join to `data_files/survey_sample_key.csv` on `id` for party, leaning, date and EMI.
Admin dashboard: `https://<vm-host>/admin`.
