from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class EnergyRiskModelBundle:
    model: object
    model_name: str
    feature_names: list[str]
    categorical_features: list[str]
    numeric_fill_values: dict[str, float]
    categorical_fill_values: dict[str, str]
    threshold: float
    native_catboost: bool = False

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError("Missing required features: " + ", ".join(missing))
        out = frame[self.feature_names].copy()
        for column, value in self.numeric_fill_values.items():
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(value)
        for column, value in self.categorical_fill_values.items():
            out[column] = out[column].astype("string").fillna(value).replace("<NA>", value).astype(str)
        return out

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.prepare(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)

