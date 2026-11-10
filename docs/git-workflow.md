# Git Workflow

Kidney Microvessel Localization использует GitHub Flow. `main` хранит стабильное
состояние проекта. Каждый change-set начинается от актуальной `main` в отдельной
ветке. Используйте префиксы `feat/`, `fix/`, `refactor/`, `docs/`, `test/` и `chore/`.
Например, `fix/validate-tile-bounds`. Для текущего процесса достаточно `main` и рабочих веток.

## Начало работы

В PowerShell из корня проекта проверьте состояние дерева и историю. Сохраните свои
незавершённые изменения в их ветке, прежде чем переключаться на `main`.

```powershell
git status --short --branch
git branch -vv
git remote -v
git log -5 --oneline --decorate
git switch main
git fetch origin
git rev-list --left-right --count main...origin/main
```

`0 0` означает, что `main` и `origin/main` совпадают. Если локальная `main` только
отстаёт, выполните `git merge --ff-only origin/main` из `main`. Стандартные
операции GitHub Flow выполняются самостоятельно в рамках согласованной задачи.
При расхождении истории остановитесь и сообщите владельцу.
Если remote ещё не подключён, владелец подключает его самостоятельно.

Когда `main` актуальна, создайте ветку задачи. Пример для документации:

```powershell
git switch -c docs/git-workflow main
uv sync --locked
```

## Проверки и коммит

Перед завершением выполните проверки проекта и дополнительные проверки по задаче.

```powershell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv build
```

Один коммит должен описывать одно связное изменение. Формат Conventional Commits
`<type>(<scope>): <summary>` допускает пропуск `scope`. Выбирайте тип по содержанию.
В сообщении укажите результат, например `fix(audit): reject invalid tile bounds`.
Добавляйте в индекс конкретные файлы и проверяйте staged diff перед коммитом.

```powershell
git add -- README.md docs/git-workflow.md
git diff --cached --check
git diff --cached
git commit -m "docs: document project Git workflow"
git log --oneline main..HEAD
git diff --stat main...HEAD
git status --short --branch
```

Храните исходные датасеты и медицинские изображения в `data/`, результаты в
`reports/`, временные файлы в `tmp/`. Секреты храните в локальных файлах, исключённых
из Git. Для подозрительного пути проверьте правило командой `git check-ignore -v -- путь`.
`.gitignore` действует на неотслеживаемые файлы, поэтому также просматривайте список
файлов в индексе. [Документация gitignore](https://git-scm.com/docs/gitignore)

## Pull Request

- Проверьте diff и имена файлов в коммитах, затем опубликуйте свою ветку командой
  `git push -u origin HEAD`. Откройте PR в `main` с названием и описанием на русском
  языке. Укажите задачу, результат, проверки и существенные ограничения.
- Дождитесь успешного завершения обязательных проверок GitHub Actions на точном
  коммите PR, включая Windows и Linux. Сверьте diff с согласованным change-set,
  проверьте автора, отсутствие конфликтов и замечаний, требующих исправления.
- Выполните squash merge собственного PR с осмысленным Conventional Commit.
  Убедитесь, что GitHub подтвердил объединение и `main` содержит ожидаемые изменения.
  Выполните `git fetch origin`, `git switch main` и `git merge --ff-only origin/main`.
  Удалите только свою завершённую ветку после проверки, что с момента ревью в неё
  не добавили коммиты. При squash исходный коммит ветки может отсутствовать среди
  предков `main`, поэтому сверяйте статус PR, итоговый коммит и содержимое diff.
- Force push, обход защиты и обязательных проверок, merge чужих PR и удаление
  репозиториев, тегов, истории и чужих веток запрещены. Настройки доступа,
  безопасности и remote меняйте по согласованию. При серьёзном архитектурном или
  методологическом противоречии сообщите владельцу перед продолжением.
- После объединения передайте краткую сводку изменений и тестов, ссылку на PR,
  хеш коммита в `main` и список оставшихся задач.

[GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) ·
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
