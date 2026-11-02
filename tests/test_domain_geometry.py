"""Проверки независимых сущностей и геометрических ограничений."""

import math
from pathlib import Path

import pytest

from kidney_vessels.annotations import validate_geometry
from kidney_vessels.domain import (
    AnnotationCompleteness,
    AnnotationStatus,
    CoordinateSystem,
    HistologyImage,
    ImageOrigin,
    Point,
    SourceCategory,
    SpatialScale,
)


def test_image_and_unknown_category_are_model_independent() -> None:
    image = HistologyImage("tile", Path("tile.tif"), 32, 16, ImageOrigin("example"))
    assert image.origin.patient_id is None
    assert image.scale is None
    assert image.coordinates == CoordinateSystem.TILE_PIXELS
    assert SourceCategory("new_structure").name == "new_structure"
    assert AnnotationStatus.EXPERT_REVIEWED.value == "expert_reviewed"
    assert AnnotationCompleteness.UNKNOWN.value == "unknown"


@pytest.mark.parametrize("width,height", [(0, 10), (10, -1)])
def test_image_requires_positive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        HistologyImage("tile", Path("tile.tif"), width, height, ImageOrigin("example"))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_point_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        Point(value, 1)


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_invalid_scale_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        SpatialScale(value, 0.5)


def test_closed_polygon_area_and_bbox() -> None:
    result = validate_geometry([[[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]], 10, 10)
    assert result.geometry is not None
    assert result.geometry.area == 12
    assert result.geometry.bbox == (1, 1, 5, 4)
    assert result.errors == ()


@pytest.mark.parametrize(
    "coordinates",
    [
        [],
        [[[1, 1], [2, 2]]],
        [[[1, 1], [2, 2], [3, 3]]],
        [[[1, 1], [5, 5], [1, 5], [5, 1]]],
        [[[-1, 1], [3, 1], [3, 3]]],
        [[[1, 1], [10, 1], [3, 3]]],
        [[[True, 1], [3, 1], [3, 3]]],
        [[[1, 1, 3], [3, 1], [3, 3]]],
        [[[math.nan, 1], [3, 1], [3, 3]]],
        "not coordinates",
    ],
)
def test_invalid_geometry_is_reported(coordinates: object) -> None:
    result = validate_geometry(coordinates, 10, 10)
    assert result.geometry is None
    assert result.errors


def test_multiple_disjoint_components_are_not_treated_as_holes() -> None:
    result = validate_geometry(
        [[[1, 1], [3, 1], [3, 3], [1, 3]], [[5, 5], [7, 5], [7, 7], [5, 7]]], 10, 10
    )
    assert result.geometry is not None
    assert result.geometry.area == 8
    assert result.geometry.bbox == (1, 1, 7, 7)


def test_duplicate_vertex_is_diagnosed_without_changing_source() -> None:
    coordinates = [[[1, 1], [3, 1], [3, 1], [3, 3], [1, 3]]]
    result = validate_geometry(coordinates, 10, 10)
    assert result.geometry is not None
    assert "repeated_vertex" in result.warnings
    assert len(coordinates[0]) == 5


def test_overflowing_area_without_known_image_is_rejected() -> None:
    result = validate_geometry([[[0, 0], [1e200, 0], [1e200, 1e200], [0, 1e200]]])
    assert result.geometry is None
    assert result.errors
