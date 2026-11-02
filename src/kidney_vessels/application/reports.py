"""Сохранение реестров, геометрической статистики и контурных примеров."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

from PIL import ImageDraw

from kidney_vessels import __version__
from kidney_vessels.data.hubmap import AnnotationRecord
from kidney_vessels.domain import HistologyImage
from kidney_vessels.preprocessing import load_rgb_preview

if TYPE_CHECKING:
    from kidney_vessels.application.audit import AuditConfig, AuditResult

COLORS = {
    "blood_vessel": (230, 30, 70),
    "glomerulus": (20, 160, 240),
    "unsure": (245, 180, 20),
}


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Записать CSV с BOM для Windows и JSON внутри составных ячеек."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _statistics(annotations: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for annotation in annotations:
        if annotation["measurable"]:
            grouped[str(annotation["category"])].append(annotation)
    rows: list[dict[str, object]] = []
    for category, entries in sorted(grouped.items()):
        for metric in ("area_px2", "bbox_width_px", "bbox_height_px", "bbox_area_px2"):
            values = [
                value for entry in entries if isinstance(value := entry[metric], (int, float))
            ]
            rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": mean(values),
                    "median": median(values),
                }
            )
    return rows


def _git_provenance() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None, None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return (
            completed.stdout.strip() if completed.returncode == 0 else None,
            bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    hashes: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        with path.open("rb") as stream:
            hashes[path.relative_to(package).as_posix()] = file_digest(stream, "sha256").hexdigest()
    return hashes


def _save_examples(
    result: AuditResult,
    config: AuditConfig,
    records: tuple[AnnotationRecord, ...],
    images: dict[str, HistologyImage],
) -> list[dict[str, object]]:
    grouped: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for record in records:
        if record.image_id is not None:
            grouped[record.image_id].append(record)
    ordered = sorted(images, key=lambda key: (not bool(grouped[key]), key))[: config.examples]
    examples: list[dict[str, object]] = []
    for index, image_id in enumerate(ordered):
        image = load_rgb_preview(images[image_id].path)
        draw = ImageDraw.Draw(image)
        count = 0
        for record in grouped[image_id]:
            for annotation in record.annotations:
                if annotation.geometry is None:
                    continue
                category = annotation.category.name if annotation.category else "unknown"
                for part in annotation.geometry.parts:
                    draw.line(
                        [(point.x, point.y) for point in (*part, part[0])],
                        fill=COLORS.get(category, (190, 70, 220)),
                        width=2,
                    )
                count += 1
        # Имя не строится из входного id: исключены обход каталогов и запрещённые символы Windows.
        relative = Path("examples") / f"example_{index:03d}.png"
        (result.output / "examples").mkdir(exist_ok=True)
        image.save(result.output / relative)
        image.close()
        for row in result.images:
            if row["image_id"] == image_id:
                row["preview"] = relative.as_posix()
        examples.append({"image_id": image_id, "path": relative.as_posix(), "drawn_objects": count})
    return examples


def save_reports(
    result: AuditResult,
    config: AuditConfig,
    records: tuple[AnnotationRecord, ...],
    tile_rows: tuple[dict[str, str], ...],
    wsi_rows: tuple[dict[str, str], ...],
    fingerprints: list[dict[str, object]],
    previews: dict[str, HistologyImage],
) -> None:
    """Сохранить согласованный набор артефактов в предварительно проверенный каталог."""
    output = result.output
    examples = _save_examples(result, config, records, previews)
    statistics = _statistics(result.annotations)
    _json(output / "summary.json", result.summary)
    _json(
        output / "registry.json",
        {
            "images": result.images,
            "annotations": result.annotations,
            "source_records": [
                {
                    "image_id": record.image_id,
                    "line": record.line,
                    "raw_line": record.raw_line,
                    "valid": record.valid,
                    "annotations": [annotation.annotation_id for annotation in record.annotations],
                }
                for record in records
            ],
            "tile_metadata": tile_rows,
            "wsi_metadata": wsi_rows,
        },
    )
    issue_rows = [asdict(issue) for issue in result.issues]
    _json(output / "issues.json", issue_rows)
    _csv(output / "images.csv", list(result.images), list(result.images[0]))
    annotation_fields = (
        list(result.annotations[0])
        if result.annotations
        else [
            "annotation_id",
            "image_id",
            "line",
            "category",
            "geometry_valid",
            "bounds_checked",
            "measurable",
            "area_px2",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
            "bbox_width_px",
            "bbox_height_px",
            "bbox_area_px2",
            "component_count",
            "errors",
            "warnings",
            "raw_json",
        ]
    )
    _csv(output / "annotations.csv", list(result.annotations), annotation_fields)
    _csv(
        output / "issues.csv",
        issue_rows,
        ["severity", "code", "message", "source", "image_id", "annotation_id", "line"],
    )
    _csv(
        output / "geometry_statistics.csv",
        statistics,
        ["category", "metric", "count", "min", "max", "mean", "median"],
    )
    _json(output / "geometry_statistics.json", statistics)
    git_revision, git_dirty = _git_provenance()
    _json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "application_version": __version__,
            "git_revision": git_revision,
            "git_dirty": git_dirty,
            "source_sha256": _source_hashes(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {name: version(name) for name in ("pillow", "shapely", "numpy")},
            "config": {
                "input": str(config.input.resolve()),
                "output": str(output),
                "examples": config.examples,
            },
            "processing": {
                "image_validation": "full_decode_single_frame",
                "coordinates": "tile_pixels_xy_top_left",
                "coordinate_bounds": "0 <= x < width; 0 <= y < height",
                "geometry": "independent_polygon_components_no_repair",
                "preview": "RGB conversion, no resize, original coordinates",
                "statistics": "geometry valid and uniquely linked readable image",
                "scale": "unknown; no unit conversion",
            },
            "input_files": fingerprints,
            "examples": examples,
            "legend_rgb": COLORS | {"unknown": (190, 70, 220)},
        },
    )
    text = [
        "# Аудит HuBMAP",
        "",
        "Этот отчёт относится только к входным файлам из manifest.json.",
        "Реальность или синтетическое происхождение изображений задаёт источник данных.",
        "",
        "## Сводка",
        "",
        "| Показатель | Значение |",
        "|---|---:|",
    ]
    for key, value in result.summary.items():
        text.append(f"| {key} | {json.dumps(value, ensure_ascii=False)} |")
    text += [
        "",
        "## Интерпретация и ограничения",
        "",
        "Dataset 1: expert_reviewed, полнота unknown; Dataset 2: not_reviewed, sparse; "
        "Dataset 3: unannotated.",
        "",
        "Отсутствие объектов не доказывает отсутствие сосудов. unsure — неопределённая "
        "исходная категория, без назначения целевой метки. Гломерулы не объединены с сосудами.",
        "",
        "Площади — непрерывные геометрические площади в пиксель²; bbox — "
        "x_min, y_min, x_max, y_max. Невалидные и несвязанные объекты исключены из статистики.",
        "",
        "Ошибки и пропущенные метаданные: [issues.csv](issues.csv). "
        "Полные записи и метаданные: [registry.json](registry.json).",
        "",
        "## Примеры",
        "",
        "Контуры: blood_vessel — красный, glomerulus — голубой, unsure — жёлтый, "
        "неизвестная категория — фиолетовый. Примеры отсортированы детерминированно.",
        "",
    ]
    for example in examples:
        # В Markdown не вставляются непроверенные идентификаторы из внешнего JSON.
        text.append(f"![Контуры]({example['path']})")
        text.append("")
    (output / "audit.md").write_text("\n".join(text) + "\n", encoding="utf-8")
