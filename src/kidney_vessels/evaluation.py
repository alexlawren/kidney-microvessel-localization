"""Попиксельная оценка бинарной сегментации в явно заданной области разметки."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import repeat
from statistics import fmean
from typing import NamedTuple


@dataclass(frozen=True)
class BinaryMask:
    """Байты 0/1 в порядке строк изображения."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or len(self.pixels) != self.width * self.height:
            raise ValueError("Размер маски не соответствует числу пикселей")
        if not set(self.pixels) <= {0, 1}:
            raise ValueError("Маска должна содержать только 0 и 1")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class MaskPair:
    """Эталон, предсказание и маска оценивания. None означает проверенную полную область."""

    reference: BinaryMask
    prediction: BinaryMask
    valid: BinaryMask | None

    def __post_init__(self) -> None:
        if self.prediction.size != self.reference.size or (
            self.valid is not None and self.valid.size != self.reference.size
        ):
            raise ValueError(
                "Эталон, предсказание и область оценивания должны совпадать по размеру"
            )


class PixelCounts(NamedTuple):
    """Счётчики внутри области оценивания."""

    true_positive: int
    false_positive: int
    false_negative: int
    evaluated_pixels: int


class Scores(NamedTuple):
    """Неопределённая метрика представлена значением None."""

    dice: float | None
    iou: float | None
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class SegmentationMetrics:
    """Итоги по тайлам и по суммарным пикселям, с явными знаменателями macro."""

    macro: Scores
    micro: Scores
    counts: PixelCounts
    evaluated_samples: int
    skipped_samples: int
    macro_defined: dict[str, int]


def _count(pair: MaskPair) -> PixelCounts:
    valid = pair.valid.pixels if pair.valid is not None else repeat(1, len(pair.reference.pixels))
    tp = fp = fn = pixels = 0
    for reference, prediction, included in zip(
        pair.reference.pixels, pair.prediction.pixels, valid, strict=True
    ):
        if included:
            pixels += 1
            tp += reference & prediction
            fp += prediction & (1 - reference)
            fn += reference & (1 - prediction)
    return PixelCounts(tp, fp, fn, pixels)


def _scores(counts: PixelCounts) -> Scores:
    tp, fp, fn, pixels = counts
    if not pixels:
        return Scores(None, None, None, None)
    return Scores(
        2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        tp / (tp + fp) if tp + fp else None,
        tp / (tp + fn) if tp + fn else None,
    )


def evaluate_segmentation(pairs: Iterable[MaskPair]) -> SegmentationMetrics:
    """Считать macro по определённым метрикам тайлов, micro по суммам TP, FP и FN."""
    per_metric: dict[str, list[float]] = {name: [] for name in Scores._fields}
    tp = fp = fn = pixels = evaluated = skipped = 0
    for pair in pairs:
        counts = _count(pair)
        if not counts.evaluated_pixels:
            skipped += 1
            continue
        evaluated += 1
        tp += counts.true_positive
        fp += counts.false_positive
        fn += counts.false_negative
        pixels += counts.evaluated_pixels
        for name, score in zip(Scores._fields, _scores(counts), strict=True):
            if score is not None:
                per_metric[name].append(score)
    totals = PixelCounts(tp, fp, fn, pixels)
    return SegmentationMetrics(
        Scores(*(fmean(values) if values else None for values in per_metric.values())),
        _scores(totals),
        totals,
        evaluated,
        skipped,
        {name: len(values) for name, values in per_metric.items()},
    )
