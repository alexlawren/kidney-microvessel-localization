"""Командная строка для Windows и других платформ."""

import argparse
import io
import logging
import sys
from pathlib import Path

from kidney_vessels.application.audit import AuditConfig, run_audit
from kidney_vessels.application.prepare import (
    CERTAIN_CATEGORIES,
    PreparationConfig,
    TargetFormat,
    run_preparation,
)
from kidney_vessels.data.hubmap import DatasetStructureError
from kidney_vessels.preparation import GeometryPolicy


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Ожидается неотрицательное целое число")
    return number


def main(argv: list[str] | None = None) -> int:
    """Запустить аудит или подготовку данных HuBMAP."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="kidney-vessels", description="Работа с данными HuBMAP")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Проверить изображения, разметку и метаданные")
    audit.add_argument("--input", type=Path, required=True, help="Каталог исходного HuBMAP")
    audit.add_argument("--output", type=Path, required=True, help="Новый или пустой каталог отчёта")
    audit.add_argument(
        "--examples", type=_nonnegative, default=6, help="Число PNG-примеров (0 отключает)"
    )
    audit.add_argument(
        "--strict", action="store_true", help="Код 1 при ошибках данных после отчёта"
    )
    audit.add_argument("--verbose", action="store_true", help="Подробное логирование")
    prepare = commands.add_parser("prepare", help="Разделить по WSI и экспортировать цели")
    prepare.add_argument("--input", required=True, type=Path, help="Исходный каталог HuBMAP")
    prepare.add_argument("--output", required=True, type=Path, help="Новый каталог экспорта")
    prepare.add_argument("--seed", default=0, type=int, help="Seed разбиения")
    prepare.add_argument(
        "--weights",
        nargs=3,
        default=(7, 2, 1),
        type=_nonnegative,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Веса разбиения по WSI",
    )
    prepare.add_argument(
        "--formats",
        nargs="+",
        choices=list(TargetFormat),
        default=list(TargetFormat),
        help="Положительные маски и/или bbox",
    )
    prepare.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CERTAIN_CATEGORIES),
        default=sorted(CERTAIN_CATEGORIES),
        help="Отбор исходных категорий",
    )
    prepare.add_argument(
        "--invalid-policy",
        choices=list(GeometryPolicy),
        default="skip",
        help="Пропуск или остановка при невалидной геометрии",
    )
    prepare.add_argument("--corrections", type=Path, help="JSON с проверенными исправлениями")
    prepare.add_argument("--limit-per-wsi", type=_nonnegative, help="Лимит экспорта на каждый WSI")
    prepare.add_argument("--wsi", nargs="+", help="Явная когорта WSI перед разбиением")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "prepare":
            manifest = run_preparation(
                PreparationConfig(
                    args.input,
                    args.output,
                    seed=args.seed,
                    weights=(args.weights[0], args.weights[1], args.weights[2]),
                    formats=frozenset(TargetFormat(value) for value in args.formats),
                    categories=frozenset(args.categories),
                    invalid_policy=GeometryPolicy(args.invalid_policy),
                    corrections=args.corrections,
                    limit_per_wsi=args.limit_per_wsi,
                    source_wsi=frozenset(args.wsi) if args.wsi else None,
                )
            )
            logging.getLogger(__name__).info("Подготовка завершена: %s", manifest)
            return 0
        result = run_audit(AuditConfig(args.input, args.output, args.examples))
    except (DatasetStructureError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("Команда %s не завершена: %s", args.command, exc)
        return 2
    return int(args.strict and any(issue.severity == "error" for issue in result.issues))
