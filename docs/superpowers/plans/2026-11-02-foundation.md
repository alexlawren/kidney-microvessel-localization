# План реализации основы и аудита HuBMAP

Цель: воспроизводимый пакет Python 3.12, читающий исходные данные и сохраняющий аудит.
Архитектура: domain → адаптеры data/annotations/preprocessing → application → CLI.
Стек: dataclasses, argparse, csv, json, Pillow, Shapely, pytest, Ruff, mypy, uv.
Выполнение последовательно в текущем новом репозитории: существующего кода нет,
поэтому отдельный worktree не требуется.

- [x] Основа: pyproject.toml, .gitignore, uv.lock; установка через `uv sync --locked`.
- [x] Тесты domain и геометрии: площадь, bbox, конечность, размер, самопересечения,
  выход за границы, замкнутый контур, многокомпонентный объект. Сначала запуск
  `uv run pytest tests/test_domain_geometry.py`, затем реализация domain/annotations.
- [x] Тесты data: проверенный контракт JSONL, CSV, пустые/отсутствующие/повреждённые
  записи, неизвестные метки, дубликаты и изображения; затем адаптеры data.
- [x] Интеграционные тесты: синтетические Dataset 1/2/3, отчёты CSV/JSON/MD/PNG,
  invalid JSON, повреждённый TIFF, orphan, отсутствие метаданных и запуск CLI.
  Затем application, preprocessing и CLI.
- [x] Документация README, architecture, dataset-audit, AGENTS; генератор synthetic.
- [x] Проверки `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy`, `uv build`; установка wheel в новое окружение и аудит synthetic.
- [x] Ревью, исправления с регрессионными тестами; Conventional Commit завершает этап.

Результат проверки 2026-11-02: 58 тестов, Ruff check/format и strict mypy проходят.
Wheel и sdist собраны. Wheel установлен без editable в новое .venv-acceptance;
runtime-зависимости установлены из экспорта uv.lock с --require-hashes.
Установленная CLI запущена из другого рабочего каталога с --strict: код 0,
3 изображения, 3 объекта, 2 WSI; артефакты CSV/JSON/Markdown/PNG созданы.
Хеши исходников установленного wheel совпадают с рабочим пакетом. PNG просмотрен.
CI для Windows/Linux настроен; удалённый CI в этой сессии не запускался.

Структура файлов: domain.py — сущности; annotations.py — геометрия;
data/{hubmap,images}.py — форматы; preprocessing.py — RGB-копии;
application/{audit,reports}.py — оркестрация/артефакты; cli.py — argparse;
tests/{conftest,test_domain_geometry,test_readers,test_audit}.py — контракты;
scripts/create_synthetic_dataset.py — автономный пример без реального датасета.
