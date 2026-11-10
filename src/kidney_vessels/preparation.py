"""Политика отбора и контролируемой коррекции исходных аннотаций."""

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from re import fullmatch

from kidney_vessels.annotations import validate_geometry
from kidney_vessels.domain import GeometricAnnotation, HistologyImage, PolygonGeometry


class GeometryPolicy(StrEnum):
    """Действие с невалидной геометрией без указанной ручной коррекции."""

    SKIP = "skip"
    ERROR = "error"


class Disposition(StrEnum):
    """Причина отбора объекта для экспорта положительной цели или пропуска."""

    TARGET = "target"
    INVALID_GEOMETRY = "invalid_geometry"
    UNSURE = "unsure"
    EXCLUDED_CATEGORY = "excluded_category"
    UNANNOTATED = "unannotated"


@dataclass(frozen=True)
class PolygonCorrection:
    """Явная замена геометрии, привязанная к неизменному исходному объекту."""

    annotation_id: str
    image_id: str
    source_sha256: str
    coordinates: object
    reason: str

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.annotation_id, self.image_id, self.reason)):
            raise ValueError("Для коррекции нужны ID аннотации, ID тайла и причина")
        if not fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("Для коррекции требуется SHA-256 исходного объекта")


@dataclass(frozen=True)
class PreparedAnnotation:
    """Исходный объект, решение подготовки и геометрия положительной цели при наличии."""

    source: GeometricAnnotation
    disposition: Disposition
    geometry: PolygonGeometry | None = None
    correction: PolygonCorrection | None = None


def annotation_sha256(annotation: GeometricAnnotation) -> str:
    """Хеш UTF-8 поля raw_json исходного объекта."""
    return sha256(annotation.raw_json.encode("utf-8")).hexdigest()


def prepare_annotation(
    annotation: GeometricAnnotation,
    image: HistologyImage,
    policy: GeometryPolicy,
    *,
    correction: PolygonCorrection | None = None,
    allowed_categories: frozenset[str] | None = None,
    allow_targets: bool = True,
) -> PreparedAnnotation:
    """Отобрать положительную цель и сохранить исходное решение валидатора."""
    if annotation.image_id != image.image_id:
        raise ValueError("Аннотация принадлежит другому изображению")
    label = annotation.category.name if annotation.category else None
    geometry = annotation.geometry
    if geometry is not None and not geometry.fits_tile(image.width, image.height):
        raise ValueError(f"Границы аннотации {annotation.annotation_id} не совпадают с тайлом")
    included = label is not None and (allowed_categories is None or label in allowed_categories)
    if correction is not None:
        if geometry is not None or not included or label == "unsure" or not allow_targets:
            raise ValueError("Коррекция допустима только для невалидной геометрии целевого объекта")
        if (correction.annotation_id, correction.image_id) != (
            annotation.annotation_id,
            annotation.image_id,
        ) or correction.source_sha256 != annotation_sha256(annotation):
            raise ValueError(f"Не совпали ID или SHA-256 коррекции {annotation.annotation_id}")
        validation = validate_geometry(correction.coordinates, image.width, image.height)
        if validation.geometry is None:
            raise ValueError(
                f"Невалидная коррекция {annotation.annotation_id}: {validation.errors}"
            )
        geometry = validation.geometry
    if geometry is None and policy == GeometryPolicy.ERROR:
        raise ValueError(f"Невалидная аннотация {annotation.annotation_id}: {annotation.errors}")
    if not allow_targets:
        return PreparedAnnotation(annotation, Disposition.UNANNOTATED)
    if label == "unsure":
        return PreparedAnnotation(annotation, Disposition.UNSURE)
    if geometry is None:
        return PreparedAnnotation(annotation, Disposition.INVALID_GEOMETRY)
    if not included:
        return PreparedAnnotation(annotation, Disposition.EXCLUDED_CATEGORY)
    return PreparedAnnotation(annotation, Disposition.TARGET, geometry, correction)
