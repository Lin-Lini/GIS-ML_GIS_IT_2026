#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _common import available_training_features, load_config, robust_clip_prob, safe_mkdir, save_json


try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:
    StratifiedGroupKFold = None


def _metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = robust_clip_prob(p)
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None, "logloss": None, "brier": None}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _build_logit(num_cols, cat_cols):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("prep", prep), ("model", model)])


def _fit_catboost(X_tr, y_tr, w_tr, X_va, y_va, cat_cols, cfg):
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        raise SystemExit("Не установлен catboost. Выполни: pip install catboost") from e
    cc = cfg["catboost"]
    vc = pd.Series(y_tr).value_counts().to_dict()
    pos = vc.get(1, 1)
    neg = vc.get(0, 1)
    pos_weight = max(1.0, neg / max(pos, 1))
    model = CatBoostClassifier(
        iterations=int(cc["iterations"]),
        learning_rate=float(cc["learning_rate"]),
        depth=int(cc["depth"]),
        l2_leaf_reg=float(cc["l2_leaf_reg"]),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=int(cc["random_seed"]),
        verbose=False,
        allow_writing_files=False,
        class_weights=[1.0, pos_weight],
    )
    fit_cols = X_tr.columns.tolist()
    cat_idx = [fit_cols.index(c) for c in cat_cols if c in fit_cols]
    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_idx,
        use_best_model=True,
        early_stopping_rounds=int(cc["early_stopping_rounds"]),
    )
    return model


def _prepare_frame(df, num_cols, cat_cols):
    out = df[num_cols + cat_cols].copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in cat_cols:
        out[c] = out[c].astype(str).fillna("NA")
    return out


def _get_splitter(X, y, groups, n_splits):
    groups = pd.Series(groups).astype(str)
    if StratifiedGroupKFold is not None and len(np.unique(y)) >= 2 and groups.nunique() >= 3:
        try:
            sgkf = StratifiedGroupKFold(n_splits=max(2, min(n_splits, groups.nunique())), shuffle=True, random_state=42)
            candidate = list(sgkf.split(X, y, groups))
            ok = True
            for tr_idx, va_idx in candidate:
                if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
                    ok = False
                    break
            if ok:
                return candidate
        except Exception:
            pass

    if groups.nunique() >= 3 and len(np.unique(y)) >= 2:
        gkf = GroupKFold(n_splits=max(2, min(n_splits, groups.nunique())))
        candidate = list(gkf.split(X, y, groups))
        ok = True
        for tr_idx, va_idx in candidate:
            if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
                ok = False
                break
        if ok:
            return candidate

    min_class = int(pd.Series(y).value_counts().min()) if len(np.unique(y)) >= 2 else 2
    skf = StratifiedKFold(n_splits=max(2, min(n_splits, min_class)), shuffle=True, random_state=42)
    return list(skf.split(X, y))


def _choose_final_model(cv_by_fold: pd.DataFrame) -> str:
    if cv_by_fold.empty:
        return "logit"
    summary = cv_by_fold.groupby("model").agg(
        pr_auc_mean=("pr_auc", "mean"),
        logloss_mean=("logloss", "mean"),
        brier_mean=("brier", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
    )
    summary = summary.loc[[m for m in ["logit", "catboost"] if m in summary.index]].copy()
    if summary.empty:
        return "logit"
    summary = summary.sort_values(["logloss_mean", "brier_mean", "pr_auc_mean", "roc_auc_mean"], ascending=[True, True, False, False])
    return str(summary.index[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = safe_mkdir(Path(args.out_dir))
    df = pd.read_csv(args.table_csv)
    num_cols, cat_cols = available_training_features(df, include_cluster=True, include_baseline=False)

    keep = df["weak_target"].isin([0, 1]) if "weak_target" in df.columns else pd.Series(False, index=df.index)
    train_df = df.loc[keep].copy()
    if len(train_df) < 50:
        raise SystemExit("Слишком мало размеченных строк для обучения")
    if train_df["weak_target"].nunique() < 2:
        raise SystemExit("В weak labels остался только один класс. Скорректируй правила labeling и перезапусти этап 02.")

    X = _prepare_frame(train_df, num_cols, cat_cols)
    y = train_df["weak_target"].astype(int).to_numpy()
    w = pd.to_numeric(train_df.get("sample_weight", 1.0), errors="coerce").fillna(1.0).to_numpy()
    groups = train_df.get("parcel_id", pd.Series(np.arange(len(train_df))))
    oof_base = train_df.get("baseline_risk", pd.Series(np.nan, index=train_df.index)).to_numpy(dtype=float)
    oof_logit = np.full(len(train_df), np.nan)
    oof_cat = np.full(len(train_df), np.nan)

    fold_rows = []
    splits = _get_splitter(X, y, groups, int(cfg["validation"]["n_splits"]))
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = w[tr_idx]

        logit = _build_logit(num_cols, cat_cols)
        logit.fit(X_tr, y_tr, model__sample_weight=w_tr)
        oof_logit[va_idx] = logit.predict_proba(X_va)[:, 1]

        cat = _fit_catboost(X_tr, y_tr, w_tr, X_va, y_va, cat_cols, cfg)
        oof_cat[va_idx] = cat.predict_proba(X_va)[:, 1]

        for name, pred in [("baseline_risk", oof_base[va_idx]), ("logit", oof_logit[va_idx]), ("catboost", oof_cat[va_idx])]:
            row = _metrics(y_va, pred)
            row.update({"fold": fold, "model": name, "n_val": int(len(va_idx))})
            fold_rows.append(row)

    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(out_dir / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    summary_df = metrics_df.groupby("model").agg(["mean", "std"])
    summary_df.to_csv(out_dir / "cv_metrics_summary.csv", encoding="utf-8-sig")

    train_df = train_df.copy()
    train_df["pred_baseline_risk_oof"] = oof_base
    train_df["pred_logit_oof"] = oof_logit
    train_df["pred_catboost_oof"] = oof_cat
    train_df["pred_ml_mean_oof"] = np.nanmean(np.vstack([oof_logit, oof_cat]), axis=0)
    train_df.to_csv(out_dir / "labeled_with_oof_predictions.csv", index=False, encoding="utf-8-sig")

    full_X = _prepare_frame(df, num_cols, cat_cols)
    full_train_X = _prepare_frame(train_df, num_cols, cat_cols)

    logit_final = _build_logit(num_cols, cat_cols)
    logit_final.fit(full_train_X, y, model__sample_weight=w)

    eval_idx = np.arange(min(200, len(full_train_X)))
    if len(np.unique(y[eval_idx])) < 2 and len(np.unique(y)) >= 2:
        pos_idx = np.where(y == 1)[0][:100]
        neg_idx = np.where(y == 0)[0][:100]
        eval_idx = np.unique(np.concatenate([pos_idx, neg_idx]))
    cat_final = _fit_catboost(full_train_X, y, w, full_train_X.iloc[eval_idx], y[eval_idx], cat_cols, cfg)

    df = df.copy()
    df["pred_logit"] = logit_final.predict_proba(full_X)[:, 1]
    df["pred_catboost"] = cat_final.predict_proba(full_X)[:, 1]
    df["pred_ml_mean"] = df[["pred_logit", "pred_catboost"]].mean(axis=1)
    if "baseline_risk" in df.columns:
        df["pred_baseline_risk"] = df["baseline_risk"]

    final_model = _choose_final_model(metrics_df)
    if final_model == "catboost":
        df["pred_ml_final"] = df["pred_catboost"]
    else:
        df["pred_ml_final"] = df["pred_logit"]
    if "pred_baseline_risk" in df.columns:
        df["pred_ml_minus_baseline"] = df["pred_ml_final"] - df["pred_baseline_risk"]

    df.to_csv(out_dir / "parcel_year_scored.csv", index=False, encoding="utf-8-sig")

    try:
        imp = cat_final.get_feature_importance(type="FeatureImportance")
        pd.DataFrame({"feature": full_train_X.columns.tolist(), "importance": imp}).sort_values("importance", ascending=False).to_csv(
            out_dir / "catboost_feature_importance.csv", index=False, encoding="utf-8-sig"
        )
    except Exception:
        pass

    joblib.dump(logit_final, out_dir / "logit_model.joblib")
    cat_final.save_model(str(out_dir / "catboost_model.cbm"))
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols, "final_model": final_model}, out_dir / "feature_schema.joblib")

    latest_mask = train_df.groupby("area")["year"].transform("max") == train_df["year"]
    if latest_mask.sum() >= 10 and train_df.loc[latest_mask, "weak_target"].nunique() == 2:
        idx_tr = np.where(~latest_mask)[0]
        idx_te = np.where(latest_mask)[0]
        X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]
        w_tr = w[idx_tr]

        logit_hold = _build_logit(num_cols, cat_cols)
        logit_hold.fit(X_tr, y_tr, model__sample_weight=w_tr)
        p_logit = logit_hold.predict_proba(X_te)[:, 1]

        cat_hold = _fit_catboost(X_tr, y_tr, w_tr, X_te, y_te, cat_cols, cfg)
        p_cat = cat_hold.predict_proba(X_te)[:, 1]
        p_base = oof_base[idx_te]
        p_mean = np.mean(np.vstack([p_logit, p_cat]), axis=0)

        pd.DataFrame([
            dict(model="baseline_risk", n_test=int(len(idx_te)), **_metrics(y_te, p_base)),
            dict(model="logit", n_test=int(len(idx_te)), **_metrics(y_te, p_logit)),
            dict(model="catboost", n_test=int(len(idx_te)), **_metrics(y_te, p_cat)),
            dict(model="ml_mean", n_test=int(len(idx_te)), **_metrics(y_te, p_mean)),
        ]).to_csv(out_dir / "holdout_latest_year_metrics.csv", index=False, encoding="utf-8-sig")

    save_json(
        out_dir / "training_report.json",
        {
            "rows_total": int(len(df)),
            "rows_labeled": int(len(train_df)),
            "class_balance": train_df["weak_target"].value_counts().to_dict(),
            "num_features": num_cols,
            "cat_features": cat_cols,
            "final_model": final_model,
            "artifacts": {
                "scored_csv": str(out_dir / "parcel_year_scored.csv"),
                "cv_summary": str(out_dir / "cv_metrics_summary.csv"),
                "oof_csv": str(out_dir / "labeled_with_oof_predictions.csv"),
                "logit_model": str(out_dir / "logit_model.joblib"),
                "catboost_model": str(out_dir / "catboost_model.cbm"),
            },
        },
    )
    print({"rows_labeled": int(len(train_df)), "final_model": final_model, "scored_csv": str(out_dir / "parcel_year_scored.csv")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
