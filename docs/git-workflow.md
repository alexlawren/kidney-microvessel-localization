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
отстаёт, владелец может обновить её командой `git merge --ff-only origin/main`.
Ассистент обновляет `main` этой командой только с явного согласия владельца.
При локальных коммитах впереди `origin/main` сначала согласуйте синхронизацию.
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

- Целевая ветка — `main`. В заголовке кратко опишите изменение, в описании укажите
  задачу, результат, проверки и существенные ограничения. Приложите ссылку на issue,
  если оно есть. Перед слиянием проверьте CI и обработайте замечания ревью.
- По завершении локальной работы передайте владельцу сводку, результаты проверок,
  название ветки и хеши коммитов. После явного согласия на публикацию выполните
  `git push -u origin HEAD` из рабочей ветки. Создание PR от имени владельца тоже согласуйте.
- Слияние в `main`, force push, удаление локальных или удалённых веток и изменение
  remote или настроек GitHub требуют явного согласия владельца.

[GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) ·
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
