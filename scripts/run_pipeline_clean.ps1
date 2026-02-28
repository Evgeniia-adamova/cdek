# Удалить старые папки с кропами лиц и запустить пайплайн
# Запуск из корня проекта:  .\scripts\run_pipeline_clean.ps1
# С видео:  .\scripts\run_pipeline_clean.ps1 data\video_from_bucket.webm

$ProjectRoot = if ($PSScriptRoot) { Split-Path $PSScriptRoot -Parent } else { Get-Location }
Set-Location $ProjectRoot

Write-Host "=== Removing old faces_by_person ===" -ForegroundColor Cyan
Remove-Item -Recurse -Force "logs\extracted_frames_v2\faces_by_person" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "logs\extracted_frames_v3_dense\faces_by_person" -ErrorAction SilentlyContinue
Write-Host "Done." -ForegroundColor Green
Write-Host ""

if ($args[0]) {
    Write-Host "=== Running pipeline with video: $($args[0]) ===" -ForegroundColor Cyan
    python -m app.run_full_pipeline $args[0]
} else {
    Write-Host "=== Running pipeline (default video) ===" -ForegroundColor Cyan
    python -m app.run_full_pipeline
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
