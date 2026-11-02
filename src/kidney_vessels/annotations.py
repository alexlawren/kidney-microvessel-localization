"""Проверка исходных полигонов без исправления или перезаписи разметки."""

import warnings as python_warnings
from dataclasses import dataclass
from math import isfinite

from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

from kidney_vessels.domain import Point, PolygonGeometry


@dataclass(frozen=True)
class GeometryValidation:
    """Результат проверки с отдельными ошибками и предупреждениями."""

    geometry: PolygonGeometry | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def validate_geometry(
    coordinates: object, width: int | None = None, height: int | None = None
) -> GeometryValidation:
    """Проверить список компонентов [[x,y], ...]; отверстия не предполагаются."""
    if not isinstance(coordinates, list) or not coordinates:
        return GeometryValidation(None, ("coordinates_not_nonempty_list",))
    parts: list[tuple[Point, ...]] = []
    warnings: set[str] = set()
    for component in coordinates:
        if not isinstance(component, list):
            return GeometryValidation(None, ("component_not_list",))
        points: list[Point] = []
        for pair in component:
            if not isinstance(pair, list) or len(pair) != 2:
                return GeometryValidation(None, ("point_not_xy_pair",))
            if not all(_finite_number(value) for value in pair):
                return GeometryValidation(None, ("coordinate_not_finite_number",))
            x, y = float(pair[0]), float(pair[1])
            if (
                x < 0
                or y < 0
                or (width is not None and x >= width)
                or (height is not None and y >= height)
            ):
                return GeometryValidation(None, ("coordinate_out_of_bounds",))
            points.append(Point(x, y))
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        if len(set(points)) < 3:
            return GeometryValidation(None, ("fewer_than_three_distinct_points",))
        if len(set(points)) != len(points):
            warnings.add("repeated_vertex")
        parts.append(tuple(points))
    try:
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("error", RuntimeWarning)
            polygons = [Polygon([(point.x, point.y) for point in part]) for part in parts]
            shape = MultiPolygon(polygons)
            if not isfinite(shape.area):
                return GeometryValidation(None, ("nonfinite_geometry_metrics",))
            if not shape.is_valid or shape.area <= 0:
                return GeometryValidation(None, (f"invalid_topology: {explain_validity(shape)}",))
    except (GEOSException, RuntimeWarning) as exc:
        return GeometryValidation(None, (f"invalid_topology: {exc}",))
    geometry = PolygonGeometry(tuple(parts))
    x_min, y_min, x_max, y_max = geometry.bbox
    if not isfinite(geometry.area) or not isfinite((x_max - x_min) * (y_max - y_min)):
        return GeometryValidation(None, ("nonfinite_geometry_metrics",))
    return GeometryValidation(geometry, warnings=tuple(sorted(warnings)))
