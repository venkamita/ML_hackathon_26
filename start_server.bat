@echo off
REM PathSnap - Laptop 1: start the backend API. Double-click or run from the repo folder.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing / checking requirements...
pip install -q -r requirements.txt
echo.
echo ===== Give this address to Laptop 2 (use the IPv4 of the Wi-Fi/hotspot adapter) =====
ipconfig | findstr /i "IPv4"
echo  -> enter  http://^<that IPv4^>:8000  in the UI sidebar
echo =======================================================================================
echo.
uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
