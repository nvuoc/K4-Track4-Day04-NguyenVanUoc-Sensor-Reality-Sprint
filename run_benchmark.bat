@echo off
chcp 65001 >nul
echo =======================================================================
echo   CHAY BENCHMARK: ADAS CAMERA DEGRADATION HEALTH SCORE (CHU DE T1)
echo =======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Khong tim thay Python tren he thong!
    pause
    exit /b 1
)

echo [1/2] Kiem tra va cai dat thu vien can thiet...
pip install -q numpy opencv-python matplotlib scipy pillow

echo [2/2] Dang chay script danh gia benchmark...
python benchmark.py

echo.
echo =======================================================================
echo   HOAN TAT! Ket qua da duoc luu tai thu muc outputs/
echo   - Bieu do: outputs/plots/
echo   - Anh truc quan: outputs/images/
echo   - Bang so lieu: outputs/results/benchmark_results.csv
echo =======================================================================
pause
