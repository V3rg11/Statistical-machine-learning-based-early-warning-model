"""Apply trained models to a separate student database."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def consecutive_prefix(row, tests):
    n = 0
    for col in tests:
        if col not in row.index or pd.isna(row[col]): break
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Scoring wide-format CSV')
    p.add_argument('--models', default='model_artifacts/models.joblib')
    p.add_argument('--config', required=True, help='Scoring JSON column mapping')
    p.add_argument('--out', default='scoring_output')
    a = p.parse_args()
    pack = joblib.load(a.models)
    cfg = json.loads(Path(a.config).read_text())
    df = pd.read_csv(a.data); tests = cfg['test_columns']
    training_tests = pack['config']['test_columns']
    if len(tests) != len(training_tests):
        raise ValueError(f'Scoring config has {len(tests)} tests but the trained model expects {len(training_tests)} in the same order.')

    position = {name: i for i, name in enumerate(training_tests)}
    missing = [c for c in tests if c not in df]
    if missing: raise ValueError(f'Missing configured scoring columns: {missing}')
    idcol, cohortcol = cfg.get('id_column'), cfg.get('cohort_column')
    for col in (idcol, cohortcol):
        if col and col not in df: raise ValueError(f'Configured column {col!r} not found')
    records = []
    for ix, row in df.iterrows():
        sid = str(row[idcol]) if idcol else str(ix)
        cohort = str(row[cohortcol]) if cohortcol else 'ALL'
        n = consecutive_prefix(row, tests)
        rec = {'student_id': sid, 'cohort': cohort, 'tests_observed': n}
        for q, item in pack['ols'].items():
            j = int(q)-1
            if n >= j+1:
                xcol = tests[position[item['x']]]; ycol = tests[position[item['y']]]
                x = float(row[xcol]); actual = float(row[ycol])
                pred = float(item['model'].predict(pd.DataFrame({item['x']:[x]}))[0])
                err = actual-pred; sd = item['resid_sd']
                rec[f'ols_pred_test_{q}'] = pred
                rec[f'ols_resid_test_{q}'] = err
                rec[f'ols_zresid_test_{q}'] = (err-item['resid_mean'])/sd if sd > 0 else np.nan
        for q, item in pack['ridge'].items():
            if n >= len(item['features']):
                X = pd.DataFrame([{c: float(row[tests[position[c]]]) for c in item['features']}])
                pred = float(item['model'].predict(X)[0])
                rec[f'ridge_pred_test_{q}'] = pred
                if n >= int(q):
                    target_col = tests[position[item['target']]]
                    actual = float(row[target_col]); rec[f'ridge_actual_test_{q}'] = actual
                    rec[f'ridge_error_test_{q}'] = actual-pred
        for stage, item in pack['knn'].items():
            s = int(stage)
            if n >= s:
                X = pd.DataFrame([{c: float(row[tests[position[c]]]) for c in item['features']}])
                distances, indices = item['nn'].kneighbors(item['scaler'].transform(X))
                ids = indices[0]; outcomes = item['outcomes'][ids]
                rec[f'knn_pred_{item["outcome"]}_at_stage_{s}'] = float(np.mean(outcomes))
                rec[f'knn_neighbor_ids_stage_{s}'] = ';'.join(item['ids'][ids].astype(str))
                rec[f'knn_neighbor_distances_stage_{s}'] = ';'.join(f'{v:.5g}' for v in distances[0])
        records.append(rec)
    result = pd.DataFrame(records)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out/'student_predictions.csv', index=False)



    summaries = []
    for col in [c for c in result if c.startswith('ols_zresid_test_')]:
        q = int(col.rsplit('_', 1)[1])
        for cohort, group in result.groupby('cohort', dropna=False):
            values = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(values):
                summaries.append({'cohort': cohort, 'test_number': q, 'n_students': len(values),
                                  'mean_standardised_residual': float(values.mean()),
                                  'sd_standardised_residual': float(values.std(ddof=1)) if len(values)>1 else np.nan,
                                  'difficulty_signal': float(values.mean())})
    pd.DataFrame(summaries).to_csv(out/'cohort_difficulty.csv', index=False)
    print(f'Wrote {out.resolve()}/student_predictions.csv and cohort_difficulty.csv')
    print('Negative difficulty_signal means this cohort underperformed historical transition expectations; it does not prove the paper was harder.')

if __name__ == '__main__': main()
