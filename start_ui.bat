@echo off
REM PathSnap - Laptop 2: start the evaluator UI. Double-click or run from the repo folder.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing / checking UI requirements...
pip install -q -r ui\requirements.txt
set /p PATHSNAP_API=Laptop 1 address (e.g. http://192.168.43.25:8000): 
streamlit run ui\app.py
pause
