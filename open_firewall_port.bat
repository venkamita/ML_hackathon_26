@echo off
REM PathSnap - Laptop 1: allow incoming connections on port 8000 (run ONCE, as Administrator).
net session >nul 2>&1
if errorlevel 1 (
    echo Please right-click this file and choose "Run as administrator".
    pause
    exit /b 1
)
netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
echo Port 8000 is open.
pause
