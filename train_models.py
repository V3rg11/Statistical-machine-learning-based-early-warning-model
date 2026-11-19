"""Train OLS transition, Ridge endpoint and KNN trajectory artefacts."""
import argparse, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def folds(n, requested):
    return KFold(n_splits=min(n, requested), shuffle=True, random_state=42) if n >= 3 else None


def ridge_equation(pipe, features):
    scaler, model = pipe.named_steps['standardscaler'], pipe.named_steps['ridge']
    coefs = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    return 'y = ' + f'{intercept:.5g}' + ''.join(f' + ({b:.5g} * {name})' for b, name in zip(coefs, features))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Historical wide-format CSV, one row per student')
    p.add_argument('--config', required=True, help='Training JSON configuration')
    p.add_argument('--out', default='model_artifacts')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tests = cfg['test_columns']
    if len(tests) < 2 or len(set(tests)) != len(tests):
        raise ValueError('test_columns must contain at least two unique, ordered column names')
    df = pd.read_csv(a.data)
    missing = [c for c in tests if c not in df]
    if missing: raise ValueError(f'Missing configured test columns: {missing}')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pack = {'config': cfg, 'ols': {}, 'ridge': {}, 'knn': {}}
    report = []


    if cfg.get('train_ols_transitions', True):
        for j in range(1, len(tests)):
            xcol, ycol = tests[j-1], tests[j]
            d = df[[xcol, ycol]].dropna()
            if len(d) < 3:
                warnings.warn(f'Skipping OLS {xcol}->{ycol}: fewer than 3 complete rows'); continue
            X, y = d[[xcol]].astype(float), d[ycol].astype(float)
            model = LinearRegression().fit(X, y)
            cv = folds(len(d), int(cfg.get('cv_folds', 5)))
            residuals = y.to_numpy() - cross_val_predict(LinearRegression(), X, y, cv=cv) if cv else y.to_numpy() - model.predict(X)
            sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            mean = float(np.mean(residuals))
            key = str(j+1)
            pack['ols'][key] = {'model': model, 'x': xcol, 'y': ycol, 'resid_mean': mean, 'resid_sd': sd, 'n': len(d)}
            report.append(f'OLS {xcol} -> {ycol}: {ycol} = {model.intercept_:.6g} + {model.coef_[0]:.6g}*{xcol}; OOF residual mean={mean:.5g}, SD={sd:.5g}, n={len(d)}')


    alphas = cfg.get('ridge_alphas', [0.001, 0.01, 0.1, 1, 10, 100])
    for q in cfg.get('ridge_target_tests', [len(tests)]):
        q = int(q)
        if q < 2 or q > len(tests): raise ValueError(f'Ridge target must be 2..{len(tests)}; got {q}')
        features, target = tests[:q-1], tests[q-1]
        d = df[features + [target]].dropna()
        if len(d) < 4:
            warnings.warn(f'Skipping Ridge target {target}: fewer than 4 complete rows'); continue
        X, y = d[features].astype(float), d[target].astype(float)
        pipe = make_pipeline(StandardScaler(), Ridge())
        cv = folds(len(d), int(cfg.get('cv_folds', 5)))
        if cv:
            search = GridSearchCV(pipe, {'ridge__alpha': alphas}, scoring='neg_mean_absolute_error', cv=cv).fit(X, y)
            fitted, alpha, cvmae = search.best_estimator_, float(search.best_params_['ridge__alpha']), float(-search.best_score_)
        else:
            alpha = float(alphas[0]); fitted = pipe.set_params(ridge__alpha=alpha).fit(X, y); cvmae = None
        equation = ridge_equation(fitted, features)


        ablation = {}
        if cv and len(features) > 1:
            for feature in features:
                reduced = [c for c in features if c != feature]
                reduced_pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                scores = cross_val_score(reduced_pipe, X[reduced], y, cv=cv, scoring='neg_mean_absolute_error')
                reduced_mae = float(-scores.mean())
                ablation[feature] = {'mae_without_feature': reduced_mae,
                                     'mae_increase': reduced_mae - cvmae}
        pack['ridge'][str(q)] = {'model': fitted, 'features': features, 'target': target, 'alpha': alpha,
                                 'equation': equation, 'cv_mae': cvmae, 'ablation': ablation, 'n': len(d)}
        report.append(f'RIDGE -> {target}: {equation}; alpha={alpha:g}; CV MAE={cvmae}; n={len(d)}')
        for feature, stats in ablation.items():
            report.append(f'  Ablation {feature}: CV MAE without it={stats["mae_without_feature"]:.5g}; increase={stats["mae_increase"]:.5g}')


    kc = cfg.get('knn', {})
    if kc.get('enabled', True):
        outcome_no = int(kc.get('outcome_test', len(tests)))
        if outcome_no < 2 or outcome_no > len(tests): raise ValueError('knn.outcome_test is out of range')
        k = int(kc.get('k', 5))
        stages = kc.get('stages', list(range(1, outcome_no)))
        for stage in stages:
            stage = int(stage)
            if stage < 1 or stage >= outcome_no: continue
            features, outcome = tests[:stage], tests[outcome_no-1]
            needed = features + [outcome]
            d = df.dropna(subset=needed).copy()
            if len(d) < max(3, k):
                warnings.warn(f'Skipping KNN stage {stage}: insufficient complete rows'); continue
            scaler = StandardScaler().fit(d[features].astype(float))
            xs = scaler.transform(d[features].astype(float))
            nn = NearestNeighbors(n_neighbors=min(k, len(d))).fit(xs)
            idcol = cfg.get('id_column')
            ids = d[idcol].astype(str).to_numpy() if idcol and idcol in d else d.index.astype(str).to_numpy()
            pack['knn'][str(stage)] = {'scaler': scaler, 'nn': nn, 'features': features, 'outcomes': d[outcome].astype(float).to_numpy(), 'ids': ids, 'outcome': outcome, 'k': min(k, len(d))}
            report.append(f'KNN stage {stage} -> {outcome}: features={features}, k={min(k,len(d))}, reference n={len(d)}')

    if not any(pack[k] for k in ('ols', 'ridge', 'knn')): raise RuntimeError('No models trained; inspect completeness/configuration')
    joblib.dump(pack, out/'models.joblib')
    metadata = {'test_columns': tests, 'ols_targets': list(pack['ols']), 'ridge_targets': list(pack['ridge']), 'knn_stages': list(pack['knn']), 'config': cfg}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    (out/'equations_and_report.txt').write_text('\n'.join(report)+'\n')
    print(f'Trained OLS={len(pack["ols"])}, Ridge={len(pack["ridge"])}, KNN={len(pack["knn"])}. Artifacts: {out.resolve()}')

if __name__ == '__main__': main()
