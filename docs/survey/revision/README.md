# Validation surveys of the revision (potato projects)

Two new Prolific surveys that rate 300 Bundestag passages and 300 tweets on the two EMI items of the original
survey (evidence-based / intuition-based, 5-point agreement scale), 20 items per participant, 5 ratings per item,
built with `code/05_validation/survey/0561_build_survey_samples.py` and `0562_make_survey_files.py`.

| Folder | Content |
|---|---|
| `parliament/`, `twitter/` | local potato projects (ports 8001 / 8002; `potato start config.yaml -p <port>` inside the folder) |
| `parliament_pilot/` | 10-item pilot, every participant rates all items |
| `deploy/` | the same three projects configured for Prolific login on port 8080, see `deploy/README.md` |
| `original_survey_2024/` | surveyflow (consent, instructions, intro, end, gold standards) and config of the original validation survey, reused verbatim |
| `sample_summary.md` | composition of the two samples by party |

Requires `pip install potato-annotation==2.8.2`. Prolific API tokens, study ids and completion codes are
placeholders (`configs/prolific_config.yaml` is git-ignored); annotation state (`annotation_output/`,
`project.sqlite*`, `layouts/`) is not included.
