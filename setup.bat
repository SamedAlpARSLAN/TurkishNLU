@echo off
REM Tek tikla kurulum: bagimliliklari yukler ve cekirdek testleri kosar.
chcp 65001 >nul
cd /d "%~dp0"
title AIST 2026 - Kurulum

where python >nul 2>nul
if errorlevel 1 (
  echo [HATA] 'python' bulunamadi. Python 3.10+ kurun ve PATH'e ekleyin.
  pause
  exit /b 1
)

echo === Bagimliliklar yukleniyor ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [HATA] Kurulum basarisiz.
  pause
  exit /b 1
)

echo.
echo === Cekirdek testler ===
python tests\test_core.py

echo.
echo Kurulum tamam. Calistirmak icin run.bat dosyasina cift tiklayin.
pause
