#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review-requested statistics: bootstrap CI, skill scores, sMAPE, stratified MAPE,
50-iter Y-randomization, window-size sensitivity."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data", "modeling_dataset_729.csv")
_RES = os.path.join(_HERE, "..", "results")
os.makedirs(_RES, exist_ok=True)
import numpy as np, pandas as pd, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv(_DATA)
WATER = ['turb','nh3','codmn','bact','flow','doy_sin','doy_cos']
targets = {'pre': WATER, 'doseT': WATER, 'chlo': WATER + ['pre','post']}

def prep(Xtr, ytr, Xte):
    imp = KNNImputer(n_neighbors=5)
    return imp.fit_transform(Xtr), np.asarray(ytr), imp.transform(Xte)

def sliding(target, feats, win=120, step=5, n_est=100, y_override=None):
    X = df[feats].reset_index(drop=True)
    yv = df[target].values.astype(float) if y_override is None else np.asarray(y_override, dtype=float)
    m = ~pd.isna(yv)
    X = X[m].reset_index(drop=True); y = yv[m]
    n = len(X); pred = np.full(n, np.nan)
    for i in range(win, n, step):
        tr = list(range(max(0, i-win), i))
        Xtr, ytr, Xte = prep(X.iloc[tr], y[tr], X.iloc[i:i+step])
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=1)
        model.fit(Xtr, ytr)
        pred[i:i+step] = model.predict(Xte)
    mask = ~np.isnan(pred)
    return y[mask], pred[mask]

def bootstrap_r2(y, p, B=2000, seed=42):
    rng = np.random.default_rng(seed); n = len(y)
    r2s = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        r2s[b] = r2_score(y[idx], p[idx])
    return float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5)), float(r2_score(y, p))

def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    r2 = r2_score(y, p)
    rmse = mean_squared_error(y, p)**0.5
    mae = mean_absolute_error(y, p)
    mape = float(np.mean(np.abs((y-p)/np.maximum(np.abs(y),1e-6)))*100)
    smape = float(np.mean(2*np.abs(y-p)/(np.abs(y)+np.abs(p)+1e-9))*100)
    mbe = float(np.mean(p-y))
    return r2, rmse, mae, mape, smape, mbe

res = {}
for t, feats in targets.items():
    y, p = sliding(t, feats)
    r2, rmse, mae, mape, smape, mbe = metrics(y, p)
    ci_lo, ci_hi, _ = bootstrap_r2(y, p)
    # persistence aligned
    pers = np.roll(y, 1)
    pers_r2 = r2_score(y[1:], pers[1:])  # skip first (no lag)
    # skill score on the same (y[1:]) points
    yc, pc, persc = y[1:], p[1:], pers[1:]
    delta_r2 = r2_score(yc, pc) - r2_score(yc, persc)
    mase = float(mean_absolute_error(yc, pc) / max(mean_absolute_error(yc, persc), 1e-9))
    res[t] = {
        'R2': round(r2,3), 'R2_CI95': [round(ci_lo,3), round(ci_hi,3)],
        'RMSE': round(rmse,4), 'MAE': round(mae,4), 'MAPE': round(mape,1), 'sMAPE': round(smape,1), 'MBE': round(mbe,4),
        'persistence_R2': round(pers_r2,3), 'delta_R2_vs_persistence': round(delta_r2,3), 'MASE': round(mase,3),
        'n': int(len(y)),
    }
    print(t, res[t])

# dose-stratified MAPE for pre
y, p = sliding('pre', WATER)
bins = [(-1, 0.2, 'pre<0.2'), (0.2, 0.5, '0.2-0.5'), (0.5, 99, 'pre>0.5')]
strat = {}
for lo, hi, name in bins:
    sel = (y >= lo) & (y < hi)
    if sel.sum() > 0:
        strat[name] = round(float(np.mean(np.abs((y[sel]-p[sel])/np.maximum(np.abs(y[sel]),1e-6)))*100), 1)
res['pre_stratified_MAPE'] = strat
print('pre stratified MAPE:', strat)

# 50-iter global Y-randomization
def yrand50(target, feats, iters=50):
    yfull = df[target].values.astype(float)
    obs = np.where(~pd.isna(yfull))[0]
    rng = np.random.default_rng(42)
    yr = []
    for _ in range(iters):
        yperm = yfull.copy(); yperm[obs] = rng.permutation(yfull[obs])
        y2, p2 = sliding(target, feats, y_override=yperm)
        yr.append(r2_score(y2, p2))
    yr = np.array(yr)
    return float(yr.mean()), float(yr.max()), float((yr >= r2_score(y, p)).mean())
for t, feats in [('pre', WATER), ('chlo', WATER+['pre','post'])]:
    m_, mx, pv = yrand50(t, feats)
    res[f'{t}_yrand50'] = {'mean': round(m_,3), 'max': round(mx,3), 'p_frac': round(pv,3)}
    print(f'{t} yrand50:', res[f'{t}_yrand50'])

# window-size sensitivity
sens = {}
for t, feats in [('pre', WATER), ('chlo', WATER+['pre','post'])]:
    vals = {}
    for win in [90, 120, 150]:
        yw, pw = sliding(t, feats, win=win, step=10)
        vals[str(win)] = round(r2_score(yw, pw), 3)
    sens[t] = vals
    print(f'{t} window sensitivity:', vals)
res['window_sensitivity'] = sens

json.dump(res, open(os.path.join(_RES, 'results_ly0812_review.json'),'w'), indent=2)
print('saved results_ly0812_review.json')
