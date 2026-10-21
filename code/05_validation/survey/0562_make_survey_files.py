"""Write the potato files for the two EMI surveys (parliament, twitter) from the old EMI instructions.

  python docs/survey/revision/make_survey_files.py

Old survey (docs/survey/revision/emi/): intro, instruction, consent, end, rating scheme (Evidenz basiert / Intuition basiert,
5-point agreement scale). New potato structure (potato-annotation 2.8.2, potato-annotation 2.8.2 layout):
config.yaml with `phases`, jsonl pages with `description` + `annotation_type`, gold standards mixed into the stream.
The instruction text is kept verbatim (umlauts restored); only the sentence on the material differs per channel.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))

CHANNEL = {
    'parliament': dict(
        material='Auszüge aus Reden im Deutschen Bundestag (2015–2025)',
        material_short='Redeauszug', risk='Die Risiken und Unannehmlichkeiten bei der Betrachtung dieser Parlamentsreden entsprechen denen, die bei alltäglichen Aktivitäten wie dem Surfen im Internet auftreten. Es ist möglich, dass Sie auf polemische oder anstößige Aussagen stoßen.',
        hint='Die Texte sind kurze Auszüge aus längeren Reden; bewerten Sie nur den gezeigten Auszug.',
        per_user=20, minutes=15, port=8001, title='Bewertung von Auszügen aus Bundestagsreden'),
    'twitter': dict(
        material='Kurznachrichten (Tweets) von Politikerinnen und Politikern (2015–2025)',
        material_short='Tweet', risk='Die Risiken und Unannehmlichkeiten bei der Betrachtung dieser Kurznachrichten entsprechen denen, die bei alltäglichen Aktivitäten wie dem Surfen im Internet auftreten. Es ist möglich, dass Sie auf polemische oder anstößige Aussagen stoßen.',
        hint='Links und Erwähnungen (@) wurden entfernt; Hashtags (#) können Sie ignorieren.',
        per_user=20, minutes=12, port=8002, title='Bewertung von Tweets von Politikerinnen und Politikern'),
}

# ---- old texts (docs/survey/revision/emi/surveyflow), umlauts restored ---------------------------------------------
INTRO_PROZEDUR = ('Diese Studie zielt darauf ab, Annotationen politischer Diskurse zu sammeln. Der Fokus liegt hierbei auf '
                  'Evidenz-basiertem und Intuition-basiertem Sprachgebrauch. In dieser Studie werden wir den Teilnehmer*innen '
                  'politische Texte präsentieren und sie bitten, zu beurteilen, inwieweit diese auf Beweisen beruhen oder nicht.')
INTRO_EINVERSTAENDNIS = ('Durch Ihre Teilnahme an dieser Studie verstehen und stimmen Sie zu, dass der Forscher möglicherweise verpflichtet ist, '
                         'Ihr Einverständnisformular, Daten und andere persönlich identifizierbare Informationen gemäß den Anforderungen von '
                         'Gesetzen, Vorschriften, Vorladungen oder Gerichtsbeschlüssen offenzulegen. Andernfalls wird Ihre Vertraulichkeit wie '
                         'folgt gewahrt: Um Ihre Identität zu schützen, werden die Forscher folgende Schritte unternehmen: (1) Jeder Teilnehmer '
                         'erhält eine Nummer; (2) Die Forscher werden alle während der Studie gesammelten Daten nach Nummer und nicht nach Namen '
                         'aufzeichnen; (3) Alle Originalaufzeichnungen oder Daten werden an einem gesicherten Ort aufbewahrt, auf den nur '
                         'autorisierte Forscher Zugriff haben.')
INTRO_FREIWILLIG = 'Ihre Teilnahme an dieser Studie ist freiwillig. Sie können die Teilnahme jederzeit während der Forschungsaktivität abbrechen.'

INSTR = [
    ('Studie:', INTRO_PROZEDUR),
    ('Erklärung:', 'Politischer Diskurs spielt in verschiedenen Aspekten der Gesellschaft eine grundlegende Rolle, dieser kann Regierungsführungen, '
                   'Politik und die allgemeine Meinungsbildung beeinflussen. Eine Möglichkeit, Diskurs zu charakterisieren, besteht darin, seine '
                   'Abhängigkeit von Beweisen zu untersuchen. Dabei konzentrieren wir uns auf Intuition-basierte (evidenzfreie) und Evidenz-basierte '
                   'Diskurse. Diese beiden Konstrukte schließen sich nicht unbedingt gegenseitig aus. Es ist möglich, dass ein Text sowohl '
                   'evidenzbasierte als auch intuitionsbasierte Passagen hat.'),
    ('Intuitionsbasierte Rhetorik:', 'Intuition-basierter Diskurs stützt sich oft auf Intuition, eine Art Bauchgefühl, Anekdoten und Meinungen. Er tendiert '
                                     'weniger zur Analyse verfügbarer Informationen und mehr zu persönlichen Überzeugungen oder emotionalen Appellen.'),
    ('Beispiel:', 'Wir müssen uns die Frage stellen, ob wir weiterhin auf kurzfristige Lösungen setzen oder ob wir den Mut haben, einen Weg zu gehen, '
                  'von dem wir alle tief im Inneren wissen, dass er langfristig das Beste für unsere Gesellschaft ist.'),
    ('Evidenzbasierte Rhetorik:', 'Evidenzbasierter Diskurs verwendet überprüfbare Fakten und Analysen. Der Schwerpunkt liegt auf der Anpassung an oder '
                                  'der Suche nach Beweisen, um zu einer fundierten Perspektive oder einem faktenbasierten Ergebnis zu gelangen.'),
    ('Beispiel:', 'Die Daten unserer umfangreichen Studien zeigen klar, dass Investitionen in erneuerbare Energien nicht nur die CO₂-Emissionen bis 2030 '
                  'um 40 % reduzieren können, sondern auch 250.000 neue Arbeitsplätze schaffen werden, wie aus dem Bericht des Bundesumweltamts hervorgeht.'),
    ('Anweisung:', 'Ihre Aufgabe ist es, für jede der unten stehenden Aussagen zu beurteilen, inwieweit es sich um einen Intuition-basierten (evidenzfreien) '
                   'oder Evidenz-basierten Diskurs handelt.'),
]
EX_INTUITION, EX_EVIDENCE = INSTR[3][1], INSTR[5][1]
OPTIONS = ['Evidenz basiert', 'Intuition basiert']
LABELS = ['Stimme überhaupt nicht zu', 'Stimme nicht zu', 'Neutral', 'Stimme zu', 'Stimme voll und ganz zu']


def jsonl(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_channel(name, c):
    d = os.path.join(HERE, name); sf = os.path.join(d, 'surveyflow'); os.makedirs(sf, exist_ok=True); os.makedirs(os.path.join(d, 'configs'), exist_ok=True)
    # intro: old "Prozedur / Risiken / Einverständnis / Freiwillige Teilnahme" as one page
    intro = (f'<h4>Name der Studie:</h4><p>{c["title"]}</p>'
             f'<h4>Prozedur:</h4><p>{INTRO_PROZEDUR} Gezeigt werden {c["material"]}.</p>'
             f'<h4>Risiken:</h4><p>{c["risk"]}</p>'
             f'<h4>Einverständnis:</h4><p>{INTRO_EINVERSTAENDNIS}</p>'
             f'<h4>Freiwillige Teilnahme:</h4><p>{INTRO_FREIWILLIG}</p>'
             f'<h4>Dauer:</h4><p>Die Teilnahme dauert etwa {c["minutes"]} Minuten.</p>'
             f'<h4>Verantwortlich:</h4><ul><li>Peer Saleth (peer.saleth@uni-konstanz.de), Universität Konstanz</li></ul>')
    jsonl(os.path.join(sf, 'intro.jsonl'), [dict(id='intro', name='intro', annotation_type='pure_display', allow_html=True, description=intro)])
    # instruction: the seven old blocks, examples set off as quotes, plus the channel hint and the scale
    parts = []
    for h, t in INSTR:
        parts.append(f'<blockquote style="margin:4px 0 12px 16px;font-style:italic">{t}</blockquote>' if h == 'Beispiel:' else f'<h4>{h}</h4><p>{t}</p>')
    parts.append(f'<h4>Hinweise:</h4><ul><li>{c["hint"]}</li><li>Beide Aussagen („Evidenz basiert“, „Intuition basiert“) werden unabhängig voneinander '
                 'auf einer Skala von „Stimme überhaupt nicht zu“ bis „Stimme voll und ganz zu“ bewertet.</li>'
                 '<li>Bewerten Sie nur, was der Text selbst ausdrückt – nicht, was Sie über die sprechende Person oder ihre Partei vermuten.</li></ul>')
    jsonl(os.path.join(sf, 'instruction.jsonl'), [dict(id='instruction', name='instruction', annotation_type='pure_display', allow_html=True, description=''.join(parts))])
    # consent: the four old yes/no items
    jsonl(os.path.join(sf, 'consent.jsonl'), [
        dict(id='consent_text', name='consent_text', annotation_type='pure_display', allow_html=True, description='<h4>Einwilligung:</h4><p>Bitte bestätigen Sie die folgenden Punkte, um an der Studie teilzunehmen.</p>'),
        dict(id='consent_read', name='consent_read', annotation_type='radio', description='Ich habe die Anweisungen gelesen und verstanden.', labels=['Ja', 'Nein'], label_requirement=dict(required=True, right_label=['Ja'])),
        dict(id='consent_age', name='consent_age', annotation_type='radio', description='Ich bin 18 Jahre oder älter.', labels=['Ja', 'Nein'], label_requirement=dict(required=True, right_label=['Ja'])),
        dict(id='consent_open', name='consent_open', annotation_type='radio', description='Ich bin einverstanden, dass meine Daten als Open Data veröffentlicht werden. Das bedeutet, dass die anonymisierten Daten öffentlich zugänglich sind und für Zwecke verwendet werden können, die nicht mit dieser Studie zusammenhängen. Es wird nicht möglich sein, Sie anhand dieser Daten zu identifizieren.', labels=['Ja', 'Nein'], label_requirement=dict(required=True, right_label=['Ja'])),
        dict(id='consent_participate', name='consent_participate', annotation_type='radio', description='Ich stimme meiner Teilnahme an der Studie zu.', labels=['Ja', 'Nein'], label_requirement=dict(required=True, right_label=['Ja'])),
    ])
    jsonl(os.path.join(sf, 'end.jsonl'), [dict(id='end', name='end', annotation_type='pure_display', allow_html=True,
                                                description='<p>Vielen Dank für Ihre Zeit.</p><p>Bei Fragen zur Studie wenden Sie sich bitte an Peer Saleth (peer.saleth@uni-konstanz.de).</p><p>Klicken Sie auf „Weiter“, um die Studie abzuschließen.</p>')])
    # gold standards (attention checks): the two instruction examples + one clearly evidence-based and one clearly intuitive item of the channel's type
    if name == 'parliament':
        g_ev = ('Nach den Zahlen des Statistischen Bundesamts ist die Zahl der Beschäftigten im vergangenen Jahr um 1,2 Prozent gestiegen; der Sachverständigenrat '
                'rechnet in seinem Gutachten für das kommende Jahr mit einem Wachstum von 0,8 Prozent. Die Evaluation des Programms durch das Institut für '
                'Arbeitsmarkt- und Berufsforschung zeigt, dass die Vermittlungsquote in den geförderten Regionen um sieben Prozentpunkte höher liegt als in den Vergleichsregionen.')
        g_iu = ('Ich glaube, jeder von uns spürt doch, dass dieser Weg der falsche ist. Man muss kein Experte sein, um zu wissen, dass das nicht gut gehen kann. '
                'Die Menschen draußen im Land haben ein feines Gespür dafür, was richtig und was falsch ist, und sie sagen uns: So geht es nicht weiter.')
    else:
        g_ev = ('Laut Statistischem Bundesamt lag die Inflationsrate im Mai bei 2,1 %. Die Studie des DIW zeigt: Haushalte mit niedrigem Einkommen sind um 0,8 Punkte '
                'stärker betroffen als der Durchschnitt. Zahlen und Quellen im verlinkten Bericht.')
        g_iu = ('Ich habe einfach das Gefühl, dass das so nicht richtig ist. Jeder, der mit offenen Augen durchs Land geht, weiß, dass es so nicht weitergehen kann. '
                'Das sagt einem doch der gesunde Menschenverstand.')
    top, bottom = LABELS[-1], LABELS[0]
    jsonl(os.path.join(sf, 'gold_standards.jsonl'), [
        dict(id='gold_evidence_example', text=EX_EVIDENCE, gold_label={'Rating': {OPTIONS[0]: top, OPTIONS[1]: bottom}}),
        dict(id='gold_intuition_example', text=EX_INTUITION, gold_label={'Rating': {OPTIONS[0]: bottom, OPTIONS[1]: top}}),
        dict(id='gold_evidence_item', text=g_ev, gold_label={'Rating': {OPTIONS[0]: top, OPTIONS[1]: bottom}}),
        dict(id='gold_intuition_item', text=g_iu, gold_label={'Rating': {OPTIONS[0]: bottom, OPTIONS[1]: top}}),
    ])
    # config
    cfg = f"""# Potato config: EMI validation survey, {name} ({c['material']})
# 300 blinded instances (data_files/survey_sample.csv, built by ../build_survey_samples.py), rated on
# "Evidenz basiert" and "Intuition basiert" with the 5-point agreement scale of the original EMI survey,
# 5 annotations per instance, {c['per_user']} instances per annotator (+ ~{c['per_user'] // 10 + 1} gold items).
#
# Local test:   potato start config.yaml -p {c['port']}     (run from this folder)
#               then open http://localhost:{c['port']}/?PROLIFIC_PID=test1
# Prolific:     switch login to type: prolific, set completion_code, fill configs/prolific_config.yaml (see README.md).

port: {c['port']}
annotation_task_name: {c['title']}
task_description: "Einschätzung, inwieweit politische Aussagen evidenzbasiert bzw. intuitionsbasiert sind. Dauer ca. {c['minutes']} Minuten."
output_annotation_dir: annotation_output/pilot/
annotation_codebook_url: ''
task_dir: .

data_files:
- data_files/survey_sample.csv
item_properties:
  id_key: id
  text_key: text
list_as_text:
  text_list_prefix_type: None
  horizontal: true

user_config:
  allow_all_users: true
  users: []

# --- Login -------------------------------------------------------------
# Local testing + simple Prolific integration: participants arrive via ?PROLIFIC_PID=<id>.
# For the full Prolific API integration switch type to "prolific" and uncomment the prolific block.
login:
  type: url_direct
  url_argument: PROLIFIC_PID
# prolific:
#   config_file_path: configs/prolific_config.yaml

# Completion code must match the code configured in the Prolific study (empty = plain thank-you page).
completion_code: ""
auto_redirect_on_completion: true
auto_redirect_delay: 5000

jumping_to_id_disabled: true
hide_navbar: true

# --- Study flow ----------------------------------------------------------
phases:
  order:
  - intro
  - instruction
  - consent
  - annotation
  - end
  intro:
    type: instructions
    file: surveyflow/intro.jsonl
  instruction:
    type: instructions
    file: surveyflow/instruction.jsonl
  consent:
    type: consent
    file: surveyflow/consent.jsonl
  end:
    type: poststudy
    file: surveyflow/end.jsonl

# --- Assignment ----------------------------------------------------------
# 300 instances x 5 annotations = 1,500 annotations; at {c['per_user']} per person this needs {1500 // c['per_user']} completed participants.
assignment_strategy: random
max_annotations_per_user: {c['per_user']}
max_annotations_per_item: 5
alert_time_each_instance: 10000000

# --- Quality control (attention checks) ---------------------------------------
# Known-answer items mixed into the stream (~1 per 10 instances); accuracy is tracked in the admin
# dashboard, annotators get no feedback and are not blocked.
gold_standards:
  enabled: true
  items_file: surveyflow/gold_standards.jsonl
  mode: mixed
  frequency: 10

# --- Annotation scheme (unchanged from the original EMI survey) ------------------------------
horizontal_key_bindings: true
annotation_schemes:
- annotation_type: multirate
  name: Rating
  description: Bewerten Sie diesen Text
  display_config:
    num_columns: 1
  arrangement: vertical
  options:
  - Evidenz basiert
  - Intuition basiert
  labels:
  - Stimme überhaupt nicht zu
  - Stimme nicht zu
  - Neutral
  - Stimme zu
  - Stimme voll und ganz zu
  label_requirement:
    required: true
  sequential_key_binding: true
  option_randomization: true

# --- UI ----------------------------------------------------------------
site_dir: default
export_annotation_format: csv
ui_language:
  text_to_annotate: "{c['material_short']}"
  next_button: "Weiter"
  previous_button: "Zurück"
  submit_button: "Absenden"
  progress_label: "Fortschritt"
  labeled_badge: "Bewertet"
  in_progress_badge: "In Bearbeitung"
  not_labeled_badge: "Noch nicht bewertet"
  loading: "Oberfläche wird geladen..."
  error_heading: "Fehler"
  instructions_heading: "Anleitung"
"""
    with open(os.path.join(d, 'config.yaml'), 'w', encoding='utf-8', newline='\n') as f: f.write(cfg)
    readme = f"""# EMI validation survey: {name}

Potato project ({{potato-annotation 2.8.2}}) for rating {c['material']} on the two EMI items of the original survey.

| Parameter | Value |
|---|---|
| Instances | 300 blinded texts in `data_files/survey_sample.csv` (id, text); party/EMI/date in `data_files/survey_sample_key.csv`, never shown |
| Sampling | {'AfD 80, Left 75, Greens 55, FDP 50, CDU/CSU 20, SPD 20 (oversampling the parties the original survey under-covered)' if name == 'parliament' else '50 per party (CDU/CSU, SPD, Greens, FDP, Left, AfD)'}, spread evenly over the global EMI quintiles; see `../sample_summary.md` |
| Items | Evidenz basiert, Intuition basiert (multirate grid, option order randomised) |
| Scale | Stimme überhaupt nicht zu / Stimme nicht zu / Neutral / Stimme zu / Stimme voll und ganz zu → code as −2 … +2 (or 0–4) |
| Annotations per item | 5 |
| Items per annotator | {c['per_user']}, random assignment |
| Participants needed | 300 × 5 / {c['per_user']} = **{1500 // c['per_user']}** completed |
| Attention checks | 4 gold items in `surveyflow/gold_standards.jsonl` (the two instruction examples + one evidence-based and one intuitive text), mixed in ~every 10 items |
| Flow | intro → instructions → consent → annotation → end (German, texts from the original EMI survey) |

## Run locally

```bash
cd docs/survey/revision/{name}
potato start config.yaml -p {c['port']}
```

(requires `pip install potato-annotation==2.8.2`.)

Then open <http://localhost:{c['port']}/?PROLIFIC_PID=test1> (any PID value = one annotator; a new value simulates a new participant).

Reset between test runs: stop the server and delete `annotation_output/`, `project.sqlite*`, `layouts/`.

## Output

- `annotation_output/pilot/exports/csv/annotations.csv`: one row per user × instance with `Rating.Evidenz basiert` and `Rating.Intuition basiert`.
- Rows whose `instance_id` starts with `gold_` are the attention checks; drop them from the analysis and use them to screen annotators (the admin dashboard tracks gold accuracy per user; the check is an exact label match, so treat it as a screening signal, not a hard filter).
- Join the analysis to `data_files/survey_sample_key.csv` on `id` for party, leaning, date and embedding EMI.

## Before fielding on Prolific

1. Set `completion_code:` in `config.yaml` to the code of the Prolific study.
2. Prolific external study URL: `https://<server>/?PROLIFIC_PID={{{{%PROLIFIC_PID%}}}}&STUDY_ID={{{{%STUDY_ID%}}}}&SESSION_ID={{{{%SESSION_ID%}}}}`. Screeners: German (fluent), resident in Germany, 18+.
3. Optional Prolific API integration: copy `configs/prolific_config_template.yaml` to `configs/prolific_config.yaml` (git-ignored), fill in token and study id, switch `login.type` to `prolific` and uncomment the `prolific:` block.
4. Deploy behind HTTPS (reverse proxy or `--ssl-cert/--ssl-key`).
"""
    with open(os.path.join(d, 'README.md'), 'w', encoding='utf-8', newline='\n') as f: f.write(readme)
    print('wrote', d)


if __name__ == '__main__':
    for n, c in CHANNEL.items(): write_channel(n, c)
