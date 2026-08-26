# -*- coding: utf-8 -*-
"""Box 1 列血缘审计（预处理闸门）的可运行实现 —— 论文的可迁移贡献之一。
对投加档案审计：识别“固定设定值”通道与“主动调节”通道，标记列混淆。
可直接套用到其他水厂档案。

用法: python audit_column_lineage.py [data.csv]
默认读取 ../data/modeling_dataset_729.csv（标签已校正的分析集），演示审计“通过”。
若传入 ../data/plant_operations_raw_deidentified.csv（脱敏原始档），审计会如论文 Box 1
所述标记出列混淆：原始档的 pre_dose 列实际是固定终段投量（0.10–0.20 mg/L），
并非前氧化投量——这正是列血缘审计要捕获、并已在该原始档中纠正的问题。
"""
import os, sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEF = os.path.join(_HERE, "..", "data", "modeling_dataset_729.csv")


def audit_block(s, name):
    s = s.dropna().astype(float)
    if len(s) < 2:
        return None
    rng = (float(s.min()), float(s.max()))
    std = float(s.std())
    n_uniq = int(s.nunique())
    lag1 = float(s.autocorr(1)) if len(s) > 2 else float("nan")
    return dict(name=name, range=rng, std=round(std, 4), n_unique=n_uniq, lag1=round(lag1, 3))


def column_lineage_audit(df, dose_cols):
    print("== 列血缘审计 ==")
    report = {}
    for role, col in dose_cols.items():
        if col not in df.columns:
            print(f"  [{role}] 通道 {col} 不存在，跳过")
            continue
        blk = audit_block(df[col], col)
        if blk:
            report[role] = blk
            print(f"  [{role}] {col}: range={blk['range']} std={blk['std']} "
                  f"n_unique={blk['n_unique']} lag1={blk['lag1']}")
    # 固定设定值通道：跨度（max-min）很小（论文称“essentially constant”，如终段投量 0.10–0.20 mg/L）
    fixed = [k for k, v in report.items() if (v["range"][1] - v["range"][0]) <= 0.20]
    varying = [k for k, v in report.items() if k not in fixed]
    print(f"  固定设定值通道(候选): {fixed}")
    print(f"  主动调节通道(候选): {varying}")
    confusion = False
    if "pre" in fixed or "terminal" in varying:
        confusion = True
        print("  [FLAG] 列混淆：pre 被识别为固定 / terminal 被识别为变化 —— 建模前需纠正身份")
    else:
        print("  [OK] 列身份一致：pre=变化, terminal=固定")
    return {"report": report, "fixed_channels": fixed,
            "varying_channels": varying, "column_confusion": confusion}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else _DEF
    df = pd.read_csv(path)
    column_lineage_audit(df, {"pre": "pre", "terminal": "post"})
