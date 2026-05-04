@echo off
setlocal

cd /d "%~dp0"

echo Starting backend on http://localhost:5000 ...
start "BE Flask" cmd /k "cd /d "%~dp0" && python backend\app.py"

echo Starting frontend on http://localhost:5500 ...
start "FE Static" cmd /k "cd /d "%~dp0" && python -m http.server 5500 --directory frontend"

echo.
echo Started:
echo - Backend:  http://localhost:5000/health
    echo - Frontend: http://localhost:5500/
    echo.
echo Use stop-be-fe.bat to stop both servers.

endlocal