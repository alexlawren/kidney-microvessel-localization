"""Проверки воспроизводимости и изоляции WSI."""

import pytest

from kidney_vessels.splits import Split, split_by_wsi


def test_splits_are_order_independent_and_keep_whole_wsi() -> None:
    images = {f"{wsi}-{tile}": str(wsi) for wsi in range(7) for tile in range(3)}
    split = split_by_wsi(images, seed=11)
    assert split == split_by_wsi(dict(reversed(list(images.items()))), seed=11)
    assert split != split_by_wsi(images, seed=12)
    assert set(split) == set(images)
    groups = {
        part: {images[key] for key, value in split.items() if value == part} for part in Split
    }
    assert all(groups.values())
    assert not groups[Split.TRAIN] & groups[Split.VALIDATION]
    assert not groups[Split.TEST] & (groups[Split.TRAIN] | groups[Split.VALIDATION])


def test_insufficient_or_unknown_wsi_is_reported() -> None:
    with pytest.raises(ValueError, match="WSI"):
        split_by_wsi({"a": "1", "b": "2"})
    with pytest.raises(ValueError, match="WSI"):
        split_by_wsi({"a": ""}, weights=(1, 0, 0))
    assert split_by_wsi({"a": "1"}, weights=(1, 0, 0)) == {"a": Split.TRAIN}
