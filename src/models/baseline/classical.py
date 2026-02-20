"""Classical ML baselines for knee OA classification.

Provides a small registry of scikit-learn compatible classifiers and utilities
to train/evaluate them. Optional models (xgboost/lightgbm) are included when
available without hard dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _maybe_xgboost():
    try:
        import xgboost as xgb  # type: ignore

        return xgb
    except Exception:
        return None


def _maybe_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception:
        return None


@dataclass
class TrainResult:
    name: str
    metrics: Dict[str, float]
    model: object


class BaselineClassifiers:
    @staticmethod
    def registry(random_state: int = 42) -> Dict[str, object]:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier

        models: Dict[str, object] = {
            "logreg": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,      # 增加迭代次数
                            solver="saga",
                            C=1.0,              # 默认正则化
                            class_weight="balanced",  # 处理类别不平衡
                            n_jobs=None,
                            random_state=random_state
                        ),
                    ),
                ]
            ),
            "svm_rbf": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", SVC(
                        C=1.0,                    # 降低复杂度
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",  # 处理类别不平衡
                        probability=True,
                        random_state=random_state
                    )),
                ]
            ),
            "rf": RandomForestClassifier(
                n_estimators=200,         # 减少树数量
                max_depth=10,             # 限制深度防止过拟合
                min_samples_split=10,     # 增加分割最小样本数
                min_samples_leaf=5,       # 增加叶子最小样本数
                class_weight="balanced",  # 处理类别不平衡
                n_jobs=-1,
                random_state=random_state
            ),
        }

        xgb = _maybe_xgboost()
        if xgb is not None:
            models["xgb"] = xgb.XGBClassifier(
                n_estimators=200,         # 减少树数量
                max_depth=4,              # 降低深度
                learning_rate=0.1,        # 提高学习率
                subsample=0.8,            # 降低采样率
                colsample_bytree=0.8,     # 降低特征采样率
                reg_alpha=0.1,            # L1正则化
                reg_lambda=0.1,           # L2正则化
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                verbosity=0,              # 减少输出
                random_state=random_state,
            )

        lgb = _maybe_lightgbm()
        if lgb is not None:
            models["lgbm"] = lgb.LGBMClassifier(
                n_estimators=100,      # 减少树数量
                num_leaves=15,         # 减少叶子节点
                learning_rate=0.1,     # 提高学习率
                min_child_samples=20,  # 增加最小样本数
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,         # L1正则化
                reg_lambda=0.1,        # L2正则化
                verbosity=-1,          # 减少警告输出
                random_state=random_state,
            )

        return models

    @staticmethod
    def fit_and_eval(
        name: str,
        model: object,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        from sklearn.metrics import accuracy_score, f1_score

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        metrics = {
            "acc": float(accuracy_score(y_val, y_pred)),
            "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
        }
        return TrainResult(name=name, metrics=metrics, model=model)

