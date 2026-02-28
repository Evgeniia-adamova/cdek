@echo off
REM Очистка старых папок с фото лиц и запуск пайплайна
REM Запуск: из корня проекта (cdek) выполнить:  scripts\run_pipeline_clean.bat
REM Или с указанием видео:  scripts\run_pipeline_clean.bat data\video_from_bucket.webm

set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

echo === Removing old face crop folders (faces_by_person) ===
if exist "logs\extracted_frames_v2\faces_by_person" (
    rmdir /s /q "logs\extracted_frames_v2\faces_by_person"
    echo Removed logs\extracted_frames_v2\faces_by_person
)
if exist "logs\extracted_frames_v3_dense\faces_by_person" (
    rmdir /s /q "logs\extracted_frames_v3_dense\faces_by_person"
    echo Removed logs\extracted_frames_v3_dense\faces_by_person
)
echo.

if "%~1"=="" (
    echo === Running full pipeline (using default video if present) ===
    python -m app.run_full_pipeline
) else (
    echo === Running full pipeline with video: %~1 ===
    python -m app.run_full_pipeline "%~1"
)

echo.
echo === Done. Check data\v2 and data\v3_dense for JSON; logs\* for frames and faces_by_person ===
