"""Доменные сущности без зависимостей от библиотек изображений и ML."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path


class CoordinateSystem(StrEnum):
    """Система локальных координат: начало слева сверху, x вправо, y вниз."""

    TILE_PIXELS = "tile_pixels_xy"


class AnnotationStatus(StrEnum):
    """Происхождение и экспертная проверка разметки, независимо от её полноты."""

    EXPERT_REVIEWED = "expert_reviewed"
    NOT_REVIEWED = "not_reviewed"
    UNANNOTATED = "unannotated"
    UNKNOWN = "unknown"


class AnnotationCompleteness(StrEnum):
    """Известная полнота исходной разметки."""

    SPARSE = "sparse"
    UNANNOTATED = "unannotated"
    UNKNOWN = "unknown"


class AnnotationPresence(StrEnum):
    """Фактическое состояние записи JSONL, не биологическое заключение."""

    PRESENT = "present"
    EMPTY = "empty"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass(frozen=True)
class Point:
    """Точка с конечными координатами."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("Координаты должны быть конечными числами")


@dataclass(frozen=True)
class SpatialScale:
    """Физический размер пикселя по двум осям в микрометрах."""

    micrometers_per_pixel_x: float
    micrometers_per_pixel_y: float

    def __post_init__(self) -> None:
        values = (self.micrometers_per_pixel_x, self.micrometers_per_pixel_y)
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("Масштаб должен быть конечным и положительным")


@dataclass(frozen=True)
class ImageOrigin:
    """Датасет и происхождение; WSI не подменяет идентификатор пациента."""

    dataset_name: str
    source_wsi: str | None = None
    patient_id: str | None = None
    dataset_subset: str | None = None


@dataclass(frozen=True)
class HistologyImage:
    """Описание декодированного изображения без массива конкретной библиотеки."""

    image_id: str
    path: Path
    width: int
    height: int
    origin: ImageOrigin
    coordinates: CoordinateSystem = CoordinateSystem.TILE_PIXELS
    scale: SpatialScale | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip() or self.width <= 0 or self.height <= 0:
            raise ValueError("Изображение должно иметь идентификатор и положительные размеры")


@dataclass(frozen=True)
class SourceCategory:
    """Исходное имя категории; перечень возможных категорий не фиксирован."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Имя категории не может быть пустым")


@dataclass(frozen=True)
class PolygonGeometry:
    """Проверенные независимые компоненты полигона в локальных пикселях."""

    parts: tuple[tuple[Point, ...], ...]

    def __post_init__(self) -> None:
        if not self.parts or any(len(set(part)) < 3 for part in self.parts):
            raise ValueError("Компонента полигона должна содержать три разные точки")

    @property
    def area(self) -> float:
        """Суммарная непрерывная площадь компонентов, пиксель²."""
        return sum(
            abs(sum(a.x * b.y - b.x * a.y for a, b in zip(part, (*part[1:], part[0]), strict=True)))
            / 2
            for part in self.parts
        )

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Границы x_min, y_min, x_max, y_max без дискретного расширения на пиксель."""
        points = [point for part in self.parts for point in part]
        return (
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points),
            max(point.y for point in points),
        )

    def fits_tile(self, width: int, height: int) -> bool:
        """Проверить границы уже валидированной геометрии на выбранном тайле."""
        left, top, right, bottom = self.bbox
        return 0 <= left <= right < width and 0 <= top <= bottom < height


@dataclass(frozen=True)
class GeometricAnnotation:
    """Исходный объект и его проверенное геометрическое представление при наличии."""

    annotation_id: str
    image_id: str
    category: SourceCategory | None
    geometry: PolygonGeometry | None
    raw_json: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditIssue:
    """Проблема с устойчивым кодом и привязкой к исходной записи."""

    severity: str
    code: str
    message: str
    source: str
    image_id: str | None = None
    annotation_id: str | None = None
    line: int | None = None
