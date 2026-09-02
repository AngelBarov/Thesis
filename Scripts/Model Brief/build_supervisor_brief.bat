@echo off
setlocal
cd /d "%~dp0"

where pdflatex >nul 2>&1
if errorlevel 1 (
    echo pdflatex was not found. Install MiKTeX or use Overleaf.
    pause
    exit /b 1
)

echo Compiling supervisor_model_brief.tex...
pdflatex -interaction=nonstopmode -halt-on-error supervisor_model_brief.tex
if errorlevel 1 (
    echo.
    echo Compilation failed. Review the error above.
    pause
    exit /b 1
)

pdflatex -interaction=nonstopmode -halt-on-error supervisor_model_brief.tex >nul
if errorlevel 1 (
    echo.
    echo The second compilation failed.
    pause
    exit /b 1
)

echo.
echo Success: supervisor_model_brief.pdf was rebuilt.
pause
