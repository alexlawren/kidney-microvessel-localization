"""Сегментационные метрики, области известной разметки и пустые маски."""

import pytest

from kidney_vessels.evaluation import BinaryMask, MaskPair, evaluate_segmentation


def mask(*pixels: int) -> BinaryMask:
    return BinaryMask(len(pixels), 1, bytes(pixels))


def test_macro_and_micro_use_different_denominators() -> None:
    result = evaluate_segmentation(
        [MaskPair(mask(1, 0), mask(1, 0), None), MaskPair(mask(1, 1, 1), mask(0, 0, 1), None)]
    )
    assert result.macro.dice == pytest.approx(0.75)
    assert result.micro.dice == pytest.approx(2 / 3)
    assert result.macro.iou == pytest.approx(2 / 3)
    assert result.micro.iou == pytest.approx(0.5)
    assert result.micro.precision == 1
    assert result.micro.recall == pytest.approx(0.5)


def test_unknown_pixels_are_excluded_and_unlabelled_samples_are_skipped() -> None:
    result = evaluate_segmentation(
        [
            MaskPair(mask(1, 0, 0), mask(1, 1, 0), mask(1, 0, 1)),
            MaskPair(mask(0, 0), mask(1, 1), mask(0, 0)),
        ]
    )
    assert result.micro.dice == result.micro.precision == 1
    assert result.counts.false_positive == 0
    assert result.counts.evaluated_pixels == 2
    assert result.skipped_samples == 1


def test_empty_masks_and_undefined_rates_are_distinct() -> None:
    result = evaluate_segmentation([MaskPair(mask(0), mask(0), None)])
    assert result.macro.dice == result.micro.iou == 1
    assert result.micro.precision is None and result.micro.recall is None
    assert result.macro_defined["precision"] == 0
    absent = evaluate_segmentation([MaskPair(mask(0), mask(0), mask(0))])
    assert absent.micro.dice is None and absent.macro.iou is None


def test_false_positive_and_false_negative_empty_cases() -> None:
    fp = evaluate_segmentation([MaskPair(mask(0), mask(1), None)]).micro
    assert fp.dice == fp.iou == fp.precision == 0
    assert fp.recall is None
    fn = evaluate_segmentation([MaskPair(mask(1), mask(0), None)]).micro
    assert fn.dice == fn.iou == fn.recall == 0
    assert fn.precision is None


def test_invalid_values_and_incompatible_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        BinaryMask(1, 1, b"\xff")
    with pytest.raises(ValueError):
        MaskPair(mask(0, 1), BinaryMask(1, 2, b"\x00\x01"), None)
    with pytest.raises(ValueError):
        MaskPair(mask(0), mask(0), mask(1, 0))
