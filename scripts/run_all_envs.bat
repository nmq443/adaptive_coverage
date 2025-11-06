@echo off
echo ==========================================
echo   Environment Simulation Runner
echo ==========================================
echo.
echo Choose mode:
echo [1] Run sequentially (one after another)
echo [2] Run in parallel (all at once)
echo.

set /p mode=Enter your choice (1 or 2): 

if "%mode%"=="1" (
    echo.
    echo Running all environment simulations sequentially...
    call scripts\run_env0.bat
    call scripts\run_env1.bat
    call scripts\run_env2.bat
    call scripts\run_env3.bat
    call scripts\run_env4.bat
    call scripts\run_env5.bat
    call scripts\run_env6.bat
    echo.
    echo All environments finished!
) else if "%mode%"=="2" (
    echo.
    echo Launching all environment simulations in parallel...
    start cmd /c scripts\run_env0.bat
    start cmd /c scripts\run_env1.bat
    start cmd /c scripts\run_env2.bat
    start cmd /c scripts\run_env3.bat
    start cmd /c scripts\run_env4.bat
    start cmd /c scripts\run_env5.bat
    start cmd /c scripts\run_env6.bat
    echo.
    echo All environments launched!
) else (
    echo.
    echo Invalid choice. Please run the script again.
)

pause
