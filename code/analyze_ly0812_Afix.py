#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-class fixes from the Q1-achievability report:
A1  decision-time value: dose-change-day stratified skill + lagged persistence baselines
A2  seasonal ablation: full vs season-only vs no-season configurations
B1  lab-timeline-aware (lagged) persistence baselines
B5  Diebold-Mariano significance tests vs persistence (HAC/Newey-West)
Protocol identical to analyze_ly0812_review.py (win=120, step=5, seed=42, KNN k=5)."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data", "modeling_dataset_729.csv")
_RES = os.path.join(_HERE, "..", "results")
os.makedirs(_RES, exist_ok=True)
import numpy as np, pandas as pd, json
from scipy import stats as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv(_DATA)
WATER = ['turb','nh3','codmn','bact','flow','doy_sin','doy_cos']
NOSEASON = ['turb','nh3','codmn','bact','flow']
SEASON = ['doy_sin','doy_cos']

def prep(Xtr, ytr, Xte):
    imp = KNNImputer(n_neighbors=5)
    return imp.fit_transform(Xtr), np.asarray(ytr), imp.transform(Xte)

def sliding(target, feats, win=120, step=5, n_est=100):
    """Returns obs, pred aligned on the target-observed series (contiguous test block)."""
    X = df[feats].reset_index(drop=True)
    y = df[target].values.astype(float)
    m = ~pd.isna(y)
    X = X[m].reset_index(drop=True); y = y[m]
    n = len(X); pred = np.full(n, np.nan)
    for i in range(win, n, step):
        tr = list(range(max(0, i-win), i))
        Xtr, ytr, Xte = prep(X.iloc[tr], y[tr], X.iloc[i:i+step])
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=1)
        model.fit(Xtr, ytr)
        pred[i:i+step] = model.predict(Xte)
    mask = ~np.isnan(pred)
    return y, pred, mask   # full series + prediction mask (test = indices >= 120)

def dm_test(e1, e2, lags=5):
    """Diebold-Mariano on squared-error loss differential with Newey-West HAC variance."""
    d = e1**2 - e2**2
    n = len(d); dbar = np.mean(d); dc = d - dbar
    g0 = np.dot(dc, dc)/n
    lrv = g0
    for k in range(1, lags+1):
        gk = np.dot(dc[k:], dc[:-k])/n
        lrv += 2*(1 - k/(lags+1))*gk
    dm = dbar/np.sqrt(max(lrv, 1e-18)/n)
    p = 2*(1 - st.norm.cdf(abs(dm)))
    return float(dm), float(p)

def mae(y, p):
    return float(mean_absolute_error(y, p))

res = {}

# ---------- full-model sliding (reproduction check) + A2 ablation ----------
CONFIGS = {
    'pre':  {'full': WATER, 'season_only': SEASON, 'no_season': NOSEASON},
    'chlo': {'full': WATER + ['pre','post'], 'season_only': SEASON,
             'no_season': NOSEASON + ['pre','post']},
}
for t, cfgs in CONFIGS.items():
    res[t] = {}
    y, pred, mask = sliding(t, cfgs['full'])
    test = np.where(mask)[0]
    assert test.min() >= 120
    res[t]['repro_R2'] = round(float(r2_score(y[test], pred[test])), 4)
    res[t]['n_test'] = int(len(test))
    # ablation
    ab = {}
    for name, feats in cfgs.items():
        y2, p2, m2 = sliding(t, feats)
        t2 = np.where(m2)[0]
        ab[name] = round(float(r2_score(y2[t2], p2[t2])), 3)
    ab['delta_full_minus_season_only'] = round(ab['full'] - ab['season_only'], 3)
    ab['delta_full_minus_no_season'] = round(ab['full'] - ab['no_season'], 3)
    res[t]['ablation'] = ab

    # ---------- B1: lagged persistence baselines ----------
    pers1 = y[test - 1]
    pers2 = y[test - 2]
    res[t]['persist1_R2'] = round(float(r2_score(y[test], pers1)), 3)
    res[t]['persist2_R2'] = round(float(r2_score(y[test], pers2)), 3)

    # ---------- B5: DM tests vs persistence ----------
    e_m = y[test] - pred[test]
    dm1, p1 = dm_test(e_m, y[test] - pers1)
    dm2, p2v = dm_test(e_m, y[test] - pers2)
    dm1_l1, p1_l1 = dm_test(e_m, y[test] - pers1, lags=1)
    res[t]['DM_vs_persist1'] = {'stat': round(dm1, 2), 'p': float(f'{p1:.2e}'),
                                'p_lag1': float(f'{p1_l1:.2e}')}
    res[t]['DM_vs_persist2'] = {'stat': round(dm2, 2), 'p': float(f'{p2v:.2e}')}
    res[t]['MASE_vs_persist1'] = round(mae(y[test], pred[test])/mae(y[test], pers1), 3)
    res[t]['MASE_vs_persist2'] = round(mae(y[test], pred[test])/mae(y[test], pers2), 3)

# ---------- A1: dose-change-day stratified skill (pre-oxidation dose) ----------
t = 'pre'
y, pred, mask = sliding(t, WATER)
test = np.where(mask)[0]
dy = np.abs(np.diff(y))                      # |Δy| at each position (i vs i-1)
dchg = dy[test - 1]                          # change into each test day
THR = 0.10
chg = dchg > THR
res[t]['change_day'] = {
    'threshold_mgL': THR,
    'median_abs_change': round(float(np.median(dchg)), 3),
    'mean_abs_change': round(float(np.mean(dchg)), 3),
    'n_change': int(chg.sum()), 'n_stable': int((~chg).sum()),
    'frac_change': round(float(chg.mean()), 3),
    'MAE_model_change': round(mae(y[test][chg], pred[test][chg]), 4),
    'MAE_persist1_change': round(mae(y[test][chg], y[test-1][chg]), 4),
    'MAE_model_stable': round(mae(y[test][~chg], pred[test][~chg]), 4),
    'MAE_persist1_stable': round(mae(y[test][~chg], y[test-1][~chg]), 4),
    'R2_model_change': round(float(r2_score(y[test][chg], pred[test][chg])), 3) if chg.sum() > 2 else None,
    'R2_persist1_change': round(float(r2_score(y[test][chg], y[test-1][chg])), 3) if chg.sum() > 2 else None,
}

# same stratification for chlorite (secondary)
t = 'chlo'
y, pred, mask = sliding(t, WATER + ['pre','post'])
test = np.where(mask)[0]
dy = np.abs(np.diff(y)); dchg = dy[test - 1]
THR_C = 0.05
chg = dchg > THR_C
res[t]['change_day'] = {
    'threshold_mgL': THR_C,
    'n_change': int(chg.sum()), 'frac_change': round(float(chg.mean()), 3),
    'MAE_model_change': round(mae(y[test][chg], pred[test][chg]), 4),
    'MAE_persist1_change': round(mae(y[test][chg], y[test-1][chg]), 4),
    'MAE_model_stable': round(mae(y[test][~chg], pred[test][~chg]), 4),
    'MAE_persist1_stable': round(mae(y[test][~chg], y[test-1][~chg]), 4),
}

json.dump(res, open(os.path.join(_RES, 'results_ly0812_Afix.json'), 'w'), indent=2)
print(json.dumps(res, indent=2))
