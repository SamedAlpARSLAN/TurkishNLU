@echo off
REM ============================================================
REM  Turkish NLU Segmentation (AIST 2026) - tiklanabilir menu
REM  Cift tikla calistir. Calisma dizini otomatik ayarlanir.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title AIST 2026 - Turkish NLU Segmentation

where python >nul 2>nul
if errorlevel 1 (
  echo [HATA] 'python' PATH'te bulunamadi. Python 3.10+ kurulu olmali.
  pause
  exit /b 1
)

:menu
cls
echo ============================================================
echo   AIST 2026 - Turkce NLU Segmentasyon Projesi
echo   Klasor: %CD%
echo ============================================================
echo   1) Kurulum        (pip install -r requirements.txt)
echo   2) Cekirdek test  (agsiz, hizli dogrulama)
echo   3) Veriyi indir   (MASSIVE-tr ~330MB)
echo   4) Hizalama dogrula (SS8 - 3 model x 4 strateji)
echo   5) CPU smoke test (uctan uca, indirme yok)
echo   6) TUM MATRIS koş  (GPU onerilir - CPU cok yavas)
echo   7) Sonuc tablolari (aggregate)
echo   8) Hata analizi    (bir kosum icin)
echo   A) Analizler       (korpus + tokenizer metrikleri, GPU'suz)
echo   9) Proje klasorunu ac
echo   0) Cikis
echo ------------------------------------------------------------
set /p choice="Seciminiz: "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto tests
if "%choice%"=="3" goto data
if "%choice%"=="4" goto validate
if "%choice%"=="5" goto smoke
if "%choice%"=="6" goto matrix
if "%choice%"=="7" goto aggregate
if "%choice%"=="8" goto erroran
if /i "%choice%"=="a" goto analyses
if "%choice%"=="9" goto openfolder
if "%choice%"=="0" exit /b 0
goto menu

:setup
echo.
python -m pip install -r requirements.txt
goto done

:tests
echo.
python tests\test_core.py
goto done

:data
echo.
python scripts\download_data.py
goto done

:validate
echo.
if not exist data\raw\tr-TR.jsonl (
  echo [BILGI] Veri yok, once indiriliyor...
  python scripts\download_data.py
)
python scripts\validate_alignment.py --data data\raw\tr-TR.jsonl --models berturk mbert xlmr
goto done

:smoke
echo.
python -m src.train --config configs\smoke.yaml
goto done

:matrix
echo.
echo [UYARI] Tam matris (27 kosum) GPU ister. CPU'da cok uzun surer.
set /p ok="Devam edilsin mi? (e/h): "
if /i not "%ok%"=="e" goto menu
python scripts\run_matrix.py
goto done

:aggregate
echo.
python scripts\aggregate.py
goto done

:analyses
echo.
if not exist data\raw\tr-TR.jsonl python scripts\download_data.py
python scripts\corpus_stats.py
python scripts\tokenizer_report.py --reference morphological --scope slot
python scripts\make_tables.py
echo Sonuclar: results\ klasorunde
goto done

:erroran
echo.
set /p tag="Kosum klasoru adi (orn: berturk_native_seed42): "
python scripts\error_report.py --predictions outputs\%tag%\test_predictions.json --data data\raw\tr-TR.jsonl
goto done

:openfolder
start "" "%CD%"
goto menu

:done
echo.
echo ------------------------------------------------------------
pause
goto menu
