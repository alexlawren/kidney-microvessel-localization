"""Воспроизводимое разбиение изображений по исходным WSI."""

from collections.abc import Mapping
from enum import StrEnum
from hashlib import sha256


class Split(StrEnum):
    """Назначение части выборки."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def split_by_wsi(
    wsi_by_image: Mapping[str, str], *, seed: int = 0, weights: tuple[int, int, int] = (7, 2, 1)
) -> dict[str, Split]:
    """Распределить целые группы. Веса задают доли WSI, с минимумом по одной группе."""
    if any(weight < 0 for weight in weights) or not sum(weights):
        raise ValueError("Веса выборок должны быть неотрицательными и иметь положительную сумму")
    groups = set(wsi_by_image.values())
    if any(not wsi.strip() for wsi in groups):
        raise ValueError("Для каждого изображения требуется известный WSI")
    active = [(split, weight) for split, weight in zip(Split, weights, strict=True) if weight]
    if len(groups) < len(active):
        raise ValueError("WSI недостаточно для всех частей с ненулевым весом")
    ordered = sorted(groups, key=lambda wsi: (sha256(f"{seed}\0{wsi}".encode()).digest(), wsi))
    free = len(ordered) - len(active)
    quotients = [divmod(free * weight, sum(weights)) for _, weight in active]
    sizes = [1 + quotient for quotient, _ in quotients]
    by_remainder = sorted(range(len(active)), key=lambda i: (-quotients[i][1], i))
    for index in by_remainder[: len(ordered) - sum(sizes)]:
        sizes[index] += 1
    assignments: dict[str, Split] = {}
    start = 0
    for (split, _), size in zip(active, sizes, strict=True):
        assignments.update((wsi, split) for wsi in ordered[start : start + size])
        start += size
    return {image_id: assignments[wsi] for image_id, wsi in sorted(wsi_by_image.items())}
