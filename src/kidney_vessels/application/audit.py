"""Оркестрация аудита исходных данных без постановки целевой ML-задачи."""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

from kidney_vessels.data.hubmap import (
    HUBMAP_DATASET,
    SUBSET_POLICY,
    TILE_METADATA_COLUMNS,
    AnnotationRecord,
    DatasetStructureError,
    annotation_presence,
    discover_train_entries,
    is_train_image,
    read_annotations,
    read_metadata,
)
from kidney_vessels.data.images import ImageReadError, inspect_image
from kidney_vessels.domain import (
    AnnotationCompleteness,
    AnnotationPresence,
    AnnotationStatus,
    AuditIssue,
    HistologyImage,
    ImageOrigin,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditConfig:
    """Параметры запуска, сохраняемые в manifest для воспроизводимости."""

    input: Path
    output: Path
    examples: int = 6

    def __post_init__(self) -> None:
        if self.examples < 0:
            raise ValueError("Число примеров должно быть неотрицательным")


@dataclass(frozen=True)
class AuditResult:
    """Реестры и сводка завершённого аудита."""

    output: Path
    summary: dict[str, object]
    images: tuple[dict[str, object], ...]
    annotations: tuple[dict[str, object], ...]
    issues: tuple[AuditIssue, ...]


def _check_paths(config: AuditConfig) -> list[Path]:
    root, output = config.input.resolve(), config.output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise DatasetStructureError("Каталоги input и output не должны пересекаться")
    files = discover_train_entries(root)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DatasetStructureError(f"Каталог output непуст или не является каталогом: {output}")
    return files


def _valid_offset(value: str) -> bool:
    # В исходном tile_meta.csv смещения могут содержать ведущие нули.
    digits = value.strip()
    return digits.isascii() and digits.isdecimal()


def _annotation_rows(
    records: tuple[AnnotationRecord, ...],
    dimensions: dict[str, tuple[int, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    record_counts = Counter(record.image_id for record in records)
    for record in records:
        for annotation in record.annotations:
            geometry = annotation.geometry
            bbox = geometry.bbox if geometry else None
            width = bbox[2] - bbox[0] if bbox else None
            height = bbox[3] - bbox[1] if bbox else None
            rows.append(
                {
                    "annotation_id": annotation.annotation_id,
                    "image_id": annotation.image_id,
                    "line": record.line,
                    "category": annotation.category.name if annotation.category else None,
                    "geometry_valid": geometry is not None,
                    "bounds_checked": annotation.image_id in dimensions,
                    "measurable": geometry is not None
                    and annotation.image_id in dimensions
                    and record_counts[annotation.image_id] == 1,
                    "area_px2": geometry.area if geometry else None,
                    "bbox_x_min": bbox[0] if bbox else None,
                    "bbox_y_min": bbox[1] if bbox else None,
                    "bbox_x_max": bbox[2] if bbox else None,
                    "bbox_y_max": bbox[3] if bbox else None,
                    "bbox_width_px": width,
                    "bbox_height_px": height,
                    "bbox_area_px2": width * height
                    if width is not None and height is not None
                    else None,
                    "component_count": len(geometry.parts) if geometry else None,
                    "errors": annotation.errors,
                    "warnings": annotation.warnings,
                    "raw_json": annotation.raw_json,
                }
            )
    return rows


def run_audit(config: AuditConfig) -> AuditResult:
    """Проверить набор, построить реестры и записать все артефакты аудита."""
    from kidney_vessels.application.reports import save_reports

    files = _check_paths(config)
    root, output = config.input.resolve(), config.output.resolve()
    tile_table = read_metadata(root / "tile_meta.csv", "id", TILE_METADATA_COLUMNS)
    wsi_table = read_metadata(root / "wsi_meta.csv", "source_wsi", {"source_wsi"})
    issues = [*tile_table.issues, *wsi_table.issues]
    image_files = [path for path in files if is_train_image(path)]
    ids = Counter(path.stem for path in image_files)
    dimensions: dict[str, tuple[int, int]] = {}
    images: list[dict[str, object]] = []
    previews: dict[str, HistologyImage] = {}
    fingerprints: list[dict[str, object]] = []
    for path in [root / name for name in ("polygons.jsonl", "tile_meta.csv", "wsi_meta.csv")] + [
        path for path in files if path.is_file()
    ]:
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        fingerprints.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    for path in files:
        if path not in image_files:
            issues.append(
                AuditIssue(
                    "warning",
                    "unsupported_train_entry",
                    "Элемент train не является поддерживаемым тайлом",
                    path.relative_to(root).as_posix(),
                )
            )
    for index, path in enumerate(image_files):
        image_id = path.stem
        source = path.relative_to(root).as_posix()
        metadata = tile_table.by_id.get(image_id, {})
        wsi = metadata.get("source_wsi", "").strip() or None
        subset = metadata.get("dataset", "").strip() or None
        wsi_metadata = wsi_table.by_id.get(wsi or "", {})
        patient = metadata.get("patient_id") or wsi_metadata.get("patient_id") or None
        status, completeness = SUBSET_POLICY.get(
            subset or "", (AnnotationStatus.UNKNOWN, AnnotationCompleteness.UNKNOWN)
        )
        if not metadata:
            issues.append(
                AuditIssue(
                    "warning", "missing_tile_metadata", "Нет строки tile_meta", source, image_id
                )
            )
        if not wsi:
            issues.append(
                AuditIssue(
                    "warning", "missing_source_wsi", "Неизвестен исходный WSI", source, image_id
                )
            )
        elif not wsi_metadata:
            issues.append(
                AuditIssue(
                    "warning", "missing_wsi_metadata", "Нет строки wsi_meta", source, image_id
                )
            )
        if wsi_metadata:
            for field in ("age", "sex", "race", "height", "weight", "bmi"):
                if not wsi_metadata.get(field, "").strip():
                    issues.append(
                        AuditIssue(
                            "warning",
                            "missing_wsi_field",
                            f"Не заполнено поле wsi_meta: {field}",
                            "wsi_meta.csv",
                            image_id,
                        )
                    )
        if subset not in SUBSET_POLICY:
            issues.append(
                AuditIssue(
                    "warning", "unknown_dataset_subset", "Неизвестный Dataset", source, image_id
                )
            )
        if not all(_valid_offset(metadata.get(key, "")) for key in ("i", "j")):
            issues.append(
                AuditIssue(
                    "warning",
                    "invalid_tile_offset",
                    "i/j отсутствуют или неверны",
                    source,
                    image_id,
                )
            )
        for code, message in (
            ("missing_patient_id", "Явный patient_id не предоставлен"),
            ("missing_spatial_scale", "Физический масштаб не предоставлен"),
        ):
            if code != "missing_patient_id" or patient is None:
                issues.append(AuditIssue("info", code, message, source, image_id))
        if ids[image_id] > 1:
            issues.append(
                AuditIssue(
                    "error", "duplicate_image_id", "Неоднозначный id тайла", source, image_id
                )
            )
        width: int | None = None
        height: int | None = None
        mode: str | None = None
        readable = False
        try:
            width, height, mode = inspect_image(path)
            readable = True
            if ids[image_id] == 1:
                dimensions[image_id] = (width, height)
                previews[image_id] = HistologyImage(
                    image_id,
                    path,
                    width,
                    height,
                    ImageOrigin(HUBMAP_DATASET, wsi, patient, subset),
                )
        except ImageReadError as exc:
            issues.append(AuditIssue("error", "corrupt_image", str(exc), source, image_id))
        images.append(
            {
                "image_id": image_id,
                "path": source,
                "readable": readable,
                "width": width,
                "height": height,
                "mode": mode,
                "source_wsi": wsi,
                "patient_id": patient,
                "dataset": subset,
                "annotation_status": status.value,
                "annotation_completeness": completeness.value,
                "coordinate_system": "tile_pixels_xy",
                "scale": None,
                "tile_metadata": metadata,
                "wsi_metadata": wsi_metadata,
                "preview": None,
            }
        )
        if (index + 1) % 250 == 0:
            logger.info("Проверено изображений: %d/%d", index + 1, len(image_files))
    parsed = read_annotations(root / "polygons.jsonl", dimensions)
    issues.extend(parsed.issues)
    grouped: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for record in parsed.records:
        if record.image_id is not None:
            grouped[record.image_id].append(record)
            if record.image_id not in ids:
                issues.append(
                    AuditIssue(
                        "error",
                        "orphan_annotation",
                        "Аннотация без изображения",
                        "polygons.jsonl",
                        record.image_id,
                        line=record.line,
                    )
                )
    for row in images:
        image_id = str(row["image_id"])
        records = grouped.get(image_id, [])
        presence = annotation_presence(records)
        row["annotation_presence"] = presence.value
        row["annotation_count"] = sum(len(record.annotations) for record in records)
        if row["dataset"] in {"1", "2"} and presence in {
            AnnotationPresence.MISSING,
            AnnotationPresence.EMPTY,
        }:
            issues.append(
                AuditIssue(
                    "warning",
                    "annotation_coverage_unknown",
                    "Нет объектов; отсутствие сосудов не установлено",
                    str(row["path"]),
                    image_id,
                )
            )
        if row["dataset"] == "3" and row["annotation_count"]:
            issues.append(
                AuditIssue(
                    "warning",
                    "unexpected_dataset3_annotations",
                    "У неразмеченного Dataset 3 обнаружены объекты",
                    "polygons.jsonl",
                    image_id,
                )
            )
    for image_id in sorted(tile_table.by_id.keys() - ids.keys()):
        issues.append(
            AuditIssue(
                "info",
                "metadata_without_train_image",
                "Строка CSV без train-тайла; возможен test или поднабор",
                "tile_meta.csv",
                image_id,
            )
        )
    annotations = _annotation_rows(parsed.records, dimensions)
    categories = Counter(str(row["category"]) for row in annotations if row["category"] is not None)
    summary: dict[str, object] = {
        "train_entries": len(files),
        "image_files": len(image_files),
        "readable_images": sum(bool(row["readable"]) for row in images),
        "unique_image_ids": len(ids),
        "source_wsi_count": len(
            {row["source_wsi"] for row in images if row["source_wsi"] is not None}
        ),
        "tile_metadata_rows": len(tile_table.rows),
        "wsi_metadata_rows": len(wsi_table.rows),
        "annotation_records": len(parsed.records),
        "objects": len(annotations),
        "valid_geometry_objects": sum(bool(row["geometry_valid"]) for row in annotations),
        "measurable_objects": sum(bool(row["measurable"]) for row in annotations),
        "categories": dict(sorted(categories.items())),
        "datasets": dict(sorted(Counter(str(row["dataset"]) for row in images).items())),
        "annotation_presence": dict(
            sorted(Counter(str(row["annotation_presence"]) for row in images).items())
        ),
        "issues_by_severity": dict(sorted(Counter(issue.severity for issue in issues).items())),
        "issues_by_code": dict(sorted(Counter(issue.code for issue in issues).items())),
    }
    output.mkdir(parents=True, exist_ok=True)
    result = AuditResult(output, summary, tuple(images), tuple(annotations), tuple(issues))
    save_reports(
        result, config, parsed.records, tile_table.rows, wsi_table.rows, fingerprints, previews
    )
    logger.info(
        "Аудит сохранён: %s; изображений %d, объектов %d", output, len(images), len(annotations)
    )
    return result
