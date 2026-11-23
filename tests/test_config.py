"""config.py tutarlılık testleri."""

from src import config


def test_classes_are_three() -> None:
    assert config.CLASSES == ("unripe", "ripe", "overripe")


def test_class_map_targets_valid() -> None:
    for target in config.CLASS_MAP.values():
        assert target is None or target in config.CLASSES


def test_color_vector_size_is_57() -> None:
    assert 3 * (3 + config.HIST_BINS) == 57


def test_hue_thresholds_in_opencv_range() -> None:
    for lo, hi in (config.BANANA_HUE_RANGE, config.SPOT_HUE_RANGE,
                   config.GREEN_HUE_RANGE, config.YELLOW_HUE_RANGE):
        assert 0 <= lo < hi <= 179
