@echo off
rem ============================================================
rem  Push freetoken-igpu to GitHub in one step.
rem
rem  1) Create an EMPTY repository at https://github.com/new
rem     (no README, no .gitignore, no license - all empty!)
rem  2) Edit the REPO line below: your_name/repo_name
rem  3) Run this file. Browser will ask you to log in.
rem ============================================================

set REPO=https://github.com/YOUR_NAME/YOUR_REPO.git

cd /d "%~dp0"
if "%~1" NEQ "" set REPO=%~1

echo Remote: %REPO%
git remote remove origin 2>nul
git remote add origin %REPO%
git branch -M main
git push -u origin main
if errorlevel 1 (
    echo.
    echo [!] Push failed. If asked for password - use a Personal Access Token:
    echo     github.com -^> Settings -^> Developer settings -^> Tokens (classic)
) else (
    echo.
    echo [OK] Pushed. Actions will start building automatically:
    echo     %REPO%/actions
)
pause
