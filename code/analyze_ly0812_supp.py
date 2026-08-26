#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary checks: global Y-randomization + aligned persistence baseline."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data", "modeling_dataset_729.csv")
_RES = os.path.join(_HERE, "..", "results")
os.makedirs(_RES, exist_ok=True)
import numpy as np, pandas as pd, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score

df = pd.read_csv(_DATA)
WATER = ['turb','nh3','codmn','bact','flow','doy_sin','doy_cos']
targets = {'pre': WATER, 'doseT': WATER, 'chlo': WATER + ['pre','post']}

def prep(Xtr, ytr, Xte):
    imp = KNNImputer(n_neighbors=5)
    return imp.fit_transform(Xtr), np.asarray(ytr), imp.transform(Xte)

def sliding_preds(target, feats, y_override=None, win=120, step=5, n_est=100):
    X = df[feats].reset_index(drop=True)
    yv = df[target].values.astype(float) if y_override is None else np.asarray(y_override, dtype=float)
    m = ~pd.isna(yv)
    X = X[m].reset_index(drop=True)
    y = yv[m]
    n = len(X)
    pred = np.full(n, np.nan)
    for i in range(win, n, step):
        tr = list(range(max(0, i-win), i))
        Xtr, ytr, Xte = prep(X.iloc[tr], y[tr], X.iloc[i:i+step])
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=1)
        model.fit(Xtr, ytr)
        pred[i:i+step] = model.predict(Xte)
    mask = ~np.isnan(pred)
    return y, pred, mask

out = {}
for t, feats in targets.items():
    y, pred, mask = sliding_preds(t, feats)
    yt, yp = y[mask], pred[mask]
    r2 = r2_score(yt, yp)
    # aligned persistence: predict point i with previous observed value
    pers = np.roll(y, 1)[mask]
    pers_r2 = r2_score(yt, pers)
    # global Y-randomization (permute only observed entries, keep NaN pattern)
    yfull = df[t].values.astype(float)
    obs = np.where(~pd.isna(yfull))[0]
    rng = np.random.default_rng(42)
    yr = []
    for _ in range(20):
        yperm = yfull.copy()
        yperm[obs] = rng.permutation(yfull[obs])
        y2, p2, m2 = sliding_preds(t, feats, y_override=yperm)
        yr.append(r2_score(y2[m2], p2[m2]))
    out[t] = {'sliding_R2': round(float(r2),3), 'persist_aligned_R2': round(float(pers_r2),3),
              'yrand_global_mean': round(float(np.mean(yr)),3), 'yrand_global_max': round(float(np.max(yr)),3)}
    print(t, out[t])

json.dump(out, open(os.path.join(_RES, 'results_ly0812_supp.json'),'w'), indent=2)
