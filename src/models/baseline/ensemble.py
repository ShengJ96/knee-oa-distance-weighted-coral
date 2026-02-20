"""Ensemble models (Voting, Stacking) with lazy sklearn import."""

from __future__ import annotations

from typing import Dict, List, Tuple


def _sklearn():
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    return VotingClassifier, StackingClassifier


class EnsembleModels:
    @staticmethod
    def voting(estimators: List[Tuple[str, object]], voting: str = "soft") -> object:
        VotingClassifier, _ = _sklearn()
        return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)

    @staticmethod
    def stacking(estimators: List[Tuple[str, object]], final_estimator: object | None = None) -> object:
        _, StackingClassifier = _sklearn()
        return StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)

