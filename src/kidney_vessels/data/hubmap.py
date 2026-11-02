"""Чтение проверенного исходного контракта HuBMAP Human Vasculature 2023."""

import csv
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

from kidney_vessels.annotations import validate_geometry
from kidney_vessels.domain import AuditIssue, GeometricAnnotation, SourceCategory

KNOWN_CATEGORIES = frozenset({"blood_vessel", "glomerulus", "unsure"})


class DatasetStructureError(ValueError):
    """Отсутствует обязательный файл или нарушен контракт таблицы."""


@dataclass(frozen=True)
class MetadataTable:
    """Все строки CSV и индекс первых вхождений без потери дубликатов."""

    rows: tuple[dict[str, str], ...]
    by_id: dict[str, dict[str, str]]
    issues: tuple[AuditIssue, ...]


@dataclass(frozen=True)
class AnnotationRecord:
    """Запись JSONL с оригинальным текстом, включая повреждённые записи."""

    image_id: str | None
    annotations: tuple[GeometricAnnotation, ...]
    line: int
    raw_line: str
    valid: bool


@dataclass(frozen=True)
class AnnotationReadResult:
    """Результат чтения JSONL без молчаливой потери записей."""

    records: tuple[AnnotationRecord, ...]
    issues: tuple[AuditIssue, ...]


def read_metadata(path: Path, key: str, required: set[str]) -> MetadataTable:
    """Прочитать CSV, сохранив строковые значения и диагностировав дубликаты."""
    rows: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    issues: list[AuditIssue] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            missing = required - set(header)
            if missing or len(set(header)) != len(header):
                raise DatasetStructureError(
                    f"Неверный заголовок {path.name}; отсутствуют поля: {sorted(missing)}; "
                    "имена столбцов должны быть уникальны"
                )
            for values in reader:
                line = reader.line_num
                if not values:
                    continue
                if len(values) != len(header):
                    raise DatasetStructureError(f"Неверное число полей {path.name}, строка {line}")
                row = dict(zip(header, values, strict=True))
                rows.append(row)
                identifier = row[key].strip()
                if not identifier or identifier in by_id:
                    code = "duplicate_metadata_id" if identifier else "missing_metadata_id"
                    issues.append(
                        AuditIssue(
                            "error",
                            code,
                            "Пустой или повторный ключ CSV",
                            path.name,
                            identifier or None,
                            line=line,
                        )
                    )
                    continue
                by_id[identifier] = row
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetStructureError(f"Не удалось прочитать {path}: {exc}") from exc
    return MetadataTable(tuple(rows), by_id, tuple(issues))


def _reject_constant(value: str) -> object:
    raise ValueError(f"Недопустимая JSON-константа {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"JSON-число вне конечного диапазона: {value}")
    return number


def _read_object(
    raw: object,
    image_id: str,
    line: int,
    index: int,
    dimensions: tuple[int, int] | None,
) -> GeometricAnnotation:
    annotation_id = f"line{line}:object{index}"
    raw_json = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    if not isinstance(raw, dict):
        return GeometricAnnotation(
            annotation_id, image_id, None, None, raw_json, ("annotation_not_object",)
        )
    obj = cast(dict[str, object], raw)
    label = obj.get("type")
    if not isinstance(label, str) or not label.strip():
        return GeometricAnnotation(
            annotation_id, image_id, None, None, raw_json, ("missing_category",)
        )
    validation = validate_geometry(obj.get("coordinates"), *(dimensions or (None, None)))
    warnings = list(validation.warnings)
    if label not in KNOWN_CATEGORIES:
        warnings.append("unknown_category")
    if dimensions is None:
        warnings.append("image_bounds_unchecked")
    return GeometricAnnotation(
        annotation_id,
        image_id,
        SourceCategory(label),
        validation.geometry,
        raw_json,
        validation.errors,
        tuple(warnings),
    )


def read_annotations(path: Path, dimensions: dict[str, tuple[int, int]]) -> AnnotationReadResult:
    """Читать построчно; ошибки строки/объекта не прерывают остальные записи."""
    records: list[AnnotationRecord] = []
    issues: list[AuditIssue] = []
    seen: set[str] = set()
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                image_id: str | None = None
                try:
                    raw: object = json.loads(
                        text, parse_constant=_reject_constant, parse_float=_finite_float
                    )
                    if not isinstance(raw, dict):
                        raise ValueError("Запись должна быть JSON-объектом")
                    obj = cast(dict[str, object], raw)
                    identifier = obj.get("id")
                    if not isinstance(identifier, str) or not identifier.strip():
                        raise ValueError("Поле id должно быть непустой строкой")
                    image_id = identifier
                    if image_id in seen:
                        issues.append(
                            AuditIssue(
                                "error",
                                "duplicate_annotation_record",
                                "Повторный id в JSONL; сохранены все записи",
                                path.name,
                                image_id,
                                line=line,
                            )
                        )
                    seen.add(image_id)
                    items = obj.get("annotations")
                    if not isinstance(items, list):
                        raise ValueError("Поле annotations должно быть списком")
                    annotations = tuple(
                        _read_object(item, image_id, line, index, dimensions.get(image_id))
                        for index, item in enumerate(items)
                    )
                    records.append(AnnotationRecord(image_id, annotations, line, text, True))
                    for annotation in annotations:
                        for severity, codes in (
                            ("error", annotation.errors),
                            ("warning", annotation.warnings),
                        ):
                            for code in codes:
                                issues.append(
                                    AuditIssue(
                                        severity,
                                        code.split(":")[0],
                                        code,
                                        path.name,
                                        image_id,
                                        annotation.annotation_id,
                                        line,
                                    )
                                )
                except (ValueError, OverflowError) as exc:
                    code = "invalid_json" if image_id is None else "invalid_annotation_record"
                    issues.append(
                        AuditIssue("error", code, str(exc), path.name, image_id, line=line)
                    )
                    records.append(AnnotationRecord(image_id, (), line, text, False))
    except (OSError, UnicodeError) as exc:
        raise DatasetStructureError(f"Не удалось прочитать {path}: {exc}") from exc
    return AnnotationReadResult(tuple(records), tuple(issues))
