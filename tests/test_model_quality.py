import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_saved_result_meets_the_recorded_accuracy_requirement():
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    assert metadata["selected_target_threshold_kwh"] == 200.0
    assert metadata["test_metrics"]["Accuracy"] > 0.96
    assert metadata["test_metrics"]["Balanced Accuracy"] > 0.90


def test_model_selection_did_not_use_test_data():
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    assert "validation" in metadata["selection_method"].lower()
    assert metadata["validation_date_range"][1] < metadata["test_date_range"][0]
