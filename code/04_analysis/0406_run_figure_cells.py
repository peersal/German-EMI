"""Run selected cells of analysis.ipynb outside Jupyter.
  python code/04_analysis/0406_run_figure_cells.py fig [SAMPLE_N|full] [SUFFIX]   cross-platform figure (cells 2, 5, 12, 14, 21); SAMPLE_N for a fast Twitter sample; SUFFIX for an alternative output name
  python code/04_analysis/0406_run_figure_cells.py contagion        within-sitting reaction figure + tables (contagion cell)
  python code/04_analysis/0406_run_figure_cells.py split2000        Bundestag before/after AfD entry on the 2000-2025 corpus (new cell 22)
  python code/04_analysis/0406_run_figure_cells.py split            temporal comparison tables with the configs' split_date, both datasets (cells 2, 5, 14)
  python code/04_analysis/0406_run_figure_cells.py rq1              per-dataset RQ1 figures (results/<dataset>/plots/rq1_combined) from the cached models, panel D split at the configs' split_date
  python code/04_analysis/0406_run_figure_cells.py v2               cross-platform figure version 2 (cell 25; density panels before/after the AfD's entry)
Run from code/ with the project venv.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import json, os, re, sys
os.chdir(REPO_ROOT)
nb = json.load(open('code/04_analysis/0403_analysis.ipynb', encoding='utf-8')); C = nb['cells']
def cell(i): return ''.join(C[i]['source'])
def clean(s): return '\n'.join(l for l in s.split('\n') if not l.lstrip().startswith('%'))
import matplotlib; matplotlib.use('Agg')
g = {'__name__': '__main__'}
for i in (2, 5, 14): exec(compile(clean(cell(i)), f'cell{i}', 'exec'), g)
g['plt'].show = lambda *a, **k: None
mode = sys.argv[1] if len(sys.argv) > 1 else 'fig'
if mode == 'fig':
    exec(compile(clean(cell(12)), 'cell12', 'exec'), g)
    s = clean(cell(21))
    if len(sys.argv) > 2 and sys.argv[2] != 'full': s = s.replace('SAMPLE_N = None', f'SAMPLE_N = {int(sys.argv[2])}')
    if len(sys.argv) > 3: s = s.replace('_suffix = "" if SAMPLE_N is None else "_sample"', f'_suffix = "{sys.argv[3]}"')   # alternative output suffix
    exec(compile(s, 'cell21', 'exec'), g)
elif mode == 'contagion':
    exec(compile(open(os.environ.get('CONTAGION_SRC','code/04_analysis/contagion/0411_contagion_sequence.py'), encoding='utf-8').read(), 'contagion', 'exec'), g)
elif mode == 'split2000':
    exec(compile(open('code/04_analysis/0409_xplat_split2000_cell.py', encoding='utf-8').read(), 'split2000', 'exec'), g)
elif mode == 'split':
    pd = g['pd']
    for cfg, path in [(g['parliament_config'], 'data/parliament_nhb.csv'), (g['twitter_config'], 'data/twitter_nhb.csv')]:
        name = cfg['dataset_name'].lower(); tables = os.path.join('results', name, 'tables'); os.makedirs(tables, exist_ok=True)
        df = g['load_and_prepare_data'](path, cfg['date_column'], cfg['group_column'], cfg['leaning_column'], id_column=cfg.get('id_column'), dropna_cols=cfg.get('dropna_cols'))
        res = g['temporal_comparison'](df, split_year=cfg.get('split_year', 2019), split_date=cfg.get('split_date'), methods=['ttest', 'cohens_d', 'bootstrap', 'mannwhitney'])
        pd.DataFrame([res['overall']]).to_csv(os.path.join(tables, 'temporal_comparison_overall.csv'), index=False)
        pd.DataFrame(res['by_leaning']).T.to_csv(os.path.join(tables, 'temporal_comparison_by_leaning.csv'))
        print(f'[{name}] tables written to {tables} (split {cfg.get("split_date")})')
elif mode == 'rq1':
    pd = g['pd']
    for cfg, path, key, fit in [(g['parliament_config'], 'data/parliament_nhb.csv', 'breakpoint', g['fit_breakpoint_model']),
                                (g['twitter_config'], 'data/twitter_nhb.csv', 'mixed', g['fit_mixed_model'])]:
        name = cfg['dataset_name']; plots = os.path.join('results', name.lower(), 'plots'); os.makedirs(plots, exist_ok=True)
        df = g['load_and_prepare_data'](path, cfg['date_column'], cfg['group_column'], cfg['leaning_column'], id_column=cfg.get('id_column'), dropna_cols=cfg.get('dropna_cols'))
        model = g['cache_or_fit'](name, key, fit, df, force=False)   # cache hit expected; never refits unless the cache is missing
        out = os.path.join(plots, 'rq1_combined.pdf')
        try: open(out, 'ab').close()
        except PermissionError: out = out.replace('.pdf', '_tmp.pdf'); print(f'[{name}] rq1_combined.pdf is locked, writing {out}')
        fig = g['plot_rq1'](df, model_results=model, emi_col='emi_w2vparliament', dataset_name=name, breakpoint_dates=cfg.get('breakpoint_dates'),
                            split_year=cfg.get('split_year', 2020), split_date=cfg.get('split_date'), save_path=out)
        fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white'); print(f'[{name}] saved {out} (+png), split {cfg.get("split_date")}')
elif mode == 'v2':
    exec(compile(clean(cell(12)), 'cell12', 'exec'), g)
    s = clean(cell(25))
    for a in ('results/emi_twitter_parliament_v2.pdf', 'results/emi_twitter_parliament_v2.png'):
        try: open(a, 'ab').close()
        except PermissionError: s = s.replace(a, a.replace('_v2.', '_v2_tmp.')); print(f'{a} is locked, writing a _tmp copy')
    exec(compile(s, 'cell25', 'exec'), g)
