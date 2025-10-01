@echo off
setlocal enabledelayedexpansion

REM Использование: begin_work.cmd [branch] [--clean]
REM   [branch]   - необязательное имя ветки для переключения
REM   --clean    - опционально удалить untracked файлы/папки (git clean -fd)

where git >nul 2>&1 || (echo [ERROR] Git not found in PATH. & exit /b 1)
git rev-parse --is-inside-work-tree >nul 2>&1 || (echo [ERROR] Not inside a Git repo. & exit /b 1)

REM --- Переключиться на ветку, если передана ---
set TARGET=%~1
if defined TARGET (
  if /i "%TARGET%" NEQ "--clean" (
    echo [SWITCH] git switch %TARGET%
    git switch %TARGET% || (echo [ERROR] Cannot switch to %TARGET% & exit /b 1)
    shift
  )
)

REM --- Опционально очистить untracked ---
if /i "%~1"=="--clean" (
  echo [CLEAN] git clean -fd
  git clean -fd
)

REM --- Текущая ветка ---
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set BR=%%b

REM --- Удобные настройки ---
git config --global pull.rebase true >nul
git config --global rebase.autoStash true >nul
git config --global fetch.prune true >nul

REM --- Синхронизация с удалённым ---
echo [SYNC] git fetch --all --prune
git fetch --all --prune

REM Если upstream не настроен — попытаться привязать origin/BR
git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >nul 2>&1
if errorlevel 1 (
  git ls-remote --exit-code --heads origin %BR% >nul 2>&1
  if not errorlevel 1 (
    echo [INFO] Set upstream to origin/%BR%
    git branch --set-upstream-to=origin/%BR% %BR%
  )
)

echo [SYNC] git pull --rebase
git pull --rebase
if errorlevel 1 (
  echo [WARN] Pull ^(rebase^) had issues. Resolve conflicts, then:
  echo        git add -A ^& git rebase --continue   ^|^| git commit
  exit /b 1
)

REM --- Submodules (если есть) ---
if exist .gitmodules (
  echo [SUBMODULES] git submodule update --init --recursive
  git submodule update --init --recursive
)

REM --- Git LFS (если установлен) ---
git lfs env >nul 2>&1
if not errorlevel 1 (
  echo [LFS] git lfs pull
  git lfs pull
)


echo [DONE] Begin on %COMPUTERNAME% -- branch: %BR%
endlocal
