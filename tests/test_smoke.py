"""Smoke test to verify the test harness works."""


def test_import_tracker():
    from src.tracker import BirdTracker, iou
    assert callable(iou)
    assert BirdTracker is not None


def test_import_metadata():
    from src.metadata import MetadataPrior, _date_to_period
    assert callable(_date_to_period)
    assert MetadataPrior is not None


def test_true():
    assert True
