"""Командная строка для Windows и других платформ."""

import argparse
import io
import logging
import sys
from pathlib import Path

from kidney_vessels.application.audit import AuditConfig, run_audit
from kidney_vessels.data.hubmap import DatasetStructureError


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Ожидается неотрицательное целое число")
    return number


def main(argv: list[str] | None = None) -> int:
    """Запустить аудит: 0 — отчёт, 1 — strict с ошибками данных, 2 — ошибка запуска."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="kidney-vessels", description="Аудит данных HuBMAP")
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
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_audit(AuditConfig(args.input, args.output, args.examples))
    except (DatasetStructureError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("Аудит не запущен или не завершён: %s", exc)
        return 2
    return int(args.strict and any(issue.severity == "error" for issue in result.issues))
