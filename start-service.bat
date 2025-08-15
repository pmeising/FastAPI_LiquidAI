@echo off
echo Starting LiquidAI FastAPI Service...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Building and starting the service...
docker-compose up --build

echo.
echo Service stopped.
pause
