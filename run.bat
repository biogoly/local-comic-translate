@echo off
setlocal
set "LCT_PROJECT="
if /i "%~x1"==".ctpr" set "LCT_PROJECT=%~f1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
if exist "SOURCE-INFO.json" goto setup
if not "%~1"=="" if not defined LCT_PROJECT goto setup
if exist "pyproject.toml" (
    where uv.exe >nul 2>nul
    if not errorlevel 1 goto checkout_uv
)
if exist ".venv\Scripts\python.exe" goto checkout_venv

:setup
if defined LCT_PROJECT (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\windows\start_source.ps1" "%LCT_PROJECT%"
) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\windows\start_source.ps1" %*
)
goto done

:checkout_uv
if defined LCT_PROJECT (uv run comic.py "%LCT_PROJECT%") else (uv run comic.py)
goto done

:checkout_venv
if defined LCT_PROJECT (".venv\Scripts\python.exe" comic.py "%LCT_PROJECT%") else (".venv\Scripts\python.exe" comic.py)

:done
set "LCT_EXIT=%ERRORLEVEL%"
popd
if not "%LCT_EXIT%"=="0" if not defined LCT_SOURCE_NONINTERACTIVE pause
exit /b %LCT_EXIT%
