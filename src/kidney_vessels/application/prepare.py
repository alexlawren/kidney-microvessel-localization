"""Воспроизводимый экспорт HuBMAP с положительными целями и происхождением."""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from kidney_vessels import __version__
from kidney_vessels.data.hubmap import (
    HUBMAP_DATASET,
    KNOWN_CATEGORIES,
    SUBSET_POLICY,
    TILE_METADATA_COLUMNS,
    AnnotationRecord,
    annotation_presence,
    discover_train_entries,
    is_train_image,
    read_annotations,
    read_metadata,
)
from kidney_vessels.data.images import inspect_image
from kidney_vessels.domain import (
    AnnotationCompleteness,
    AnnotationStatus,
    AuditIssue,
    HistologyImage,
    ImageOrigin,
)
from kidney_vessels.preparation import (
    GeometryPolicy,
    PolygonCorrection,
    annotation_sha256,
    prepare_annotation,
)
from kidney_vessels.preprocessing import load_rgb_preview, rasterize_polygon
from kidney_vessels.splits import Split, split_by_wsi

CERTAIN_CATEGORIES = KNOWN_CATEGORIES - {"unsure"}
logger = logging.getLogger(__name__)


class TargetFormat(StrEnum):
    """Независимые представления положительной геометрии."""

    MASK = "mask"
    BBOX = "bbox"


@dataclass(frozen=True)
class PreparationConfig:
    """Параметры подготовки, сохраняемые в манифесте."""

    input: Path
    output: Path
    seed: int = 0
    weights: tuple[int, int, int] = (7, 2, 1)
    formats: frozenset[TargetFormat] = frozenset(TargetFormat)
    categories: frozenset[str] = CERTAIN_CATEGORIES
    invalid_policy: GeometryPolicy = GeometryPolicy.SKIP
    corrections: Path | None = None
    limit_per_wsi: int | None = None
    source_wsi: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.formats or not self.categories or not self.categories <= CERTAIN_CATEGORIES:
            raise ValueError("Укажите форматы и определённые категории HuBMAP")
        if self.limit_per_wsi is not None and self.limit_per_wsi <= 0:
            raise ValueError("Лимит экспорта на WSI должен быть положительным")
        if self.source_wsi is not None and not self.source_wsi:
            raise ValueError("Когорта WSI не должна быть пустой")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _corrections(path: Path | None) -> dict[str, PolygonCorrection]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("Файл коррекций должен содержать список объектов")
    result: dict[str, PolygonCorrection] = {}
    for row in rows:
        keys = ("annotation_id", "image_id", "source_sha256", "reason")
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in keys):
            raise ValueError("У коррекции должны быть строковые ID, SHA-256 и причина")
        correction = PolygonCorrection(
            row["annotation_id"],
            row["image_id"],
            row["source_sha256"],
            row.get("coordinates"),
            row["reason"],
        )
        if correction.annotation_id in result:
            raise ValueError(f"Повторная коррекция {correction.annotation_id}")
        result[correction.annotation_id] = correction
    return result


def _object_rows(
    image: HistologyImage,
    records: list[AnnotationRecord],
    status: AnnotationStatus,
    config: PreparationConfig,
    corrections: dict[str, PolygonCorrection],
    directory: Path,
    index: int,
) -> list[dict[str, object]]:
    ambiguous = len(records) > 1 or any(not record.valid for record in records)
    if ambiguous and config.invalid_policy == GeometryPolicy.ERROR:
        raise ValueError(f"Неоднозначная запись JSONL для тайла {image.image_id}")
    rows: list[dict[str, object]] = []
    for record in records:
        for annotation in record.annotations:
            bbox: tuple[float, float, float, float] | None = None
            mask_path = None
            correction = None
            if ambiguous:
                disposition = "ambiguous_record"
            else:
                prepared = prepare_annotation(
                    annotation,
                    image,
                    config.invalid_policy,
                    correction=corrections.pop(annotation.annotation_id, None),
                    allowed_categories=config.categories,
                    allow_targets=status != AnnotationStatus.UNANNOTATED,
                )
                disposition = prepared.disposition.value
                correction = prepared.correction
                if prepared.geometry is not None:
                    if TargetFormat.BBOX in config.formats:
                        bbox = prepared.geometry.bbox
                    if TargetFormat.MASK in config.formats:
                        mask_path = f"masks/{index:06d}_{len(rows):04d}.png"
                        with rasterize_polygon(
                            prepared.geometry, (image.width, image.height)
                        ) as mask:
                            mask.save(directory / mask_path)
            rows.append(
                {
                    "annotation_id": annotation.annotation_id,
                    "line": record.line,
                    "category": annotation.category.name if annotation.category else None,
                    "disposition": disposition,
                    "source_json": annotation.raw_json,
                    "source_sha256": annotation_sha256(annotation),
                    "source_errors": annotation.errors,
                    "source_warnings": annotation.warnings,
                    "correction": asdict(correction) if correction is not None else None,
                    "bbox_xyxy": bbox,
                    "mask": mask_path,
                }
            )
    return rows


def run_preparation(config: PreparationConfig) -> Path:
    """Записать каталог подготовки атомарно после проверки всех выбранных тайлов."""
    root, output = config.input.resolve(), config.output.resolve()
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Каталоги input и output не должны пересекаться")
    if output.exists():
        raise ValueError("Для подготовки требуется новый каталог output")
    files = [path for path in discover_train_entries(root) if is_train_image(path)]
    image_paths = {path.stem: path for path in files}
    if len(image_paths) != len(files):
        raise ValueError("Неоднозначные идентификаторы train-тайлов")
    tiles = read_metadata(root / "tile_meta.csv", "id", TILE_METADATA_COLUMNS)
    slides = read_metadata(root / "wsi_meta.csv", "source_wsi", {"source_wsi"})
    if tiles.issues or slides.issues:
        raise ValueError("Исправьте неоднозначные ключи метаданных до подготовки")
    wsi_by_image = {
        image_id: tiles.by_id.get(image_id, {}).get("source_wsi", "").strip()
        for image_id in image_paths
    }
    if any(not wsi for wsi in wsi_by_image.values()):
        raise ValueError("Для каждого изображения требуется известный WSI")
    if config.source_wsi is not None:
        unknown = config.source_wsi - set(wsi_by_image.values())
        if unknown:
            raise ValueError(f"Неизвестные WSI: {', '.join(sorted(unknown))}")
        wsi_by_image = {
            image_id: wsi for image_id, wsi in wsi_by_image.items() if wsi in config.source_wsi
        }
    assignments = split_by_wsi(wsi_by_image, seed=config.seed, weights=config.weights)
    group_counts: Counter[str] = Counter()
    dimensions: dict[str, tuple[int, int]] = {}
    for image_id in assignments:
        wsi = wsi_by_image[image_id]
        if config.limit_per_wsi is not None and group_counts[wsi] >= config.limit_per_wsi:
            continue
        width, height, _ = inspect_image(image_paths[image_id])
        dimensions[image_id] = (width, height)
        group_counts[wsi] += 1
    parsed = read_annotations(
        root / "polygons.jsonl", dimensions, skip_image_ids=image_paths.keys() - dimensions.keys()
    )
    records: dict[str, list[AnnotationRecord]] = defaultdict(list)
    unlinked: list[AnnotationRecord] = []
    issues = list(parsed.issues)
    for record in parsed.records:
        if record.image_id is not None and record.image_id in dimensions:
            records[record.image_id].append(record)
        else:
            unlinked.append(record)
            if record.image_id is not None:
                issues.append(
                    AuditIssue(
                        "error",
                        "orphan_annotation",
                        "Аннотация не связана с train-тайлом",
                        "polygons.jsonl",
                        record.image_id,
                        line=record.line,
                    )
                )
    if unlinked and config.invalid_policy == GeometryPolicy.ERROR:
        raise ValueError("В JSONL найдена запись без соответствующего train-тайла")
    corrections = _corrections(config.corrections)
    source_hashes = {
        name: _sha256(root / name) for name in ("tile_meta.csv", "wsi_meta.csv", "polygons.jsonl")
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as temporary:
        directory = Path(temporary) / "prepared"
        (directory / "images").mkdir(parents=True)
        if TargetFormat.MASK in config.formats:
            (directory / "masks").mkdir()
        rows: list[dict[str, object]] = []
        exported: Counter[Split] = Counter()
        with_targets: Counter[Split] = Counter()
        target_groups: dict[Split, set[str]] = defaultdict(set)
        for index, (image_id, (width, height)) in enumerate(dimensions.items()):
            tile = tiles.by_id[image_id]
            wsi = wsi_by_image[image_id]
            slide = slides.by_id.get(wsi, {})
            subset = tile["dataset"].strip()
            status, completeness = SUBSET_POLICY.get(
                subset, (AnnotationStatus.UNKNOWN, AnnotationCompleteness.UNKNOWN)
            )
            patient = (tile.get("patient_id") or slide.get("patient_id") or "").strip() or None
            image = HistologyImage(
                image_id,
                image_paths[image_id],
                width,
                height,
                ImageOrigin(HUBMAP_DATASET, wsi, patient, subset or None),
            )
            original_records = records.get(image_id, [])
            objects = _object_rows(
                image, original_records, status, config, corrections, directory, index
            )
            part = assignments[image_id]
            has_targets = any(row["disposition"] == "target" for row in objects)
            exported[part] += 1
            if has_targets:
                with_targets[part] += 1
                target_groups[part].add(wsi)
            rgb_path = f"images/{index:06d}.png"
            with load_rgb_preview(image.path) as rgb:
                rgb.save(directory / rgb_path)
            rows.append(
                {
                    "image_id": image_id,
                    "partition": assignments[image_id],
                    "path": rgb_path,
                    "source_path": image.path.relative_to(root).as_posix(),
                    "source_sha256": _sha256(image.path),
                    "origin": asdict(image.origin),
                    "width": width,
                    "height": height,
                    "coordinate_system": image.coordinates,
                    "scale": asdict(image.scale) if image.scale else None,
                    "tile_metadata": tile,
                    "wsi_metadata": slide,
                    "annotation_status": status,
                    "annotation_completeness": completeness,
                    "annotation_presence": annotation_presence(original_records),
                    "target_coverage": "positive_only" if has_targets else "unknown",
                    "source_records": [
                        {"line": record.line, "raw_line": record.raw_line}
                        for record in original_records
                    ],
                    "objects": objects,
                }
            )
        if corrections:
            raise ValueError(f"Неиспользованные коррекции: {', '.join(sorted(corrections))}")
        for part in Split:
            if exported[part] and not with_targets[part]:
                logger.warning("В экспортированной части %s нет положительных целей", part)
        manifest = {
            "schema_version": 1,
            "application_version": __version__,
            "libraries": {name: version(name) for name in ("pillow", "shapely")},
            "input": str(root),
            "source_sha256": source_hashes,
            "parameters": {
                "seed": config.seed,
                "weights": config.weights,
                "formats": sorted(config.formats),
                "categories": sorted(config.categories),
                "invalid_policy": config.invalid_policy,
                "limit_per_wsi": config.limit_per_wsi,
                "source_wsi": sorted(config.source_wsi) if config.source_wsi else None,
                "corrections_sha256": _sha256(config.corrections) if config.corrections else None,
                "image_transform": "RGB conversion; no resize",
                "rasterization": "Pillow ImageDraw.polygon, mask values 0 and 255",
            },
            "patient_independence": "not_verified",
            "split_coverage": {
                part: {
                    "exported_tiles": exported[part],
                    "exported_tiles_with_targets": with_targets[part],
                    "wsi_with_exported_targets": sorted(target_groups[part]),
                }
                for part in Split
            },
            "splits": {
                part: dict(
                    Counter(
                        wsi_by_image[image_id]
                        for image_id, partition in assignments.items()
                        if partition == part
                    )
                )
                for part in Split
            },
            "assignments": [
                {
                    "image_id": key,
                    "source_wsi": wsi_by_image[key],
                    "partition": part,
                    "dataset": tiles.by_id[key]["dataset"],
                    "exported": key in dimensions,
                }
                for key, part in assignments.items()
            ],
            "images": rows,
            "issues": [asdict(issue) for issue in issues],
            "unlinked_records": [
                {"image_id": record.image_id, "line": record.line, "raw_line": record.raw_line}
                for record in unlinked
            ],
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise ValueError("Каталог output появился во время подготовки")
        directory.rename(output)
    return output / "manifest.json"
