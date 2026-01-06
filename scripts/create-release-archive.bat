@echo off
REM Script to create release archive for etx-tracer
REM Creates a zip containing bin/ contents (except assets_testing, lib, and tmp) plus blender plugin in blender/ folder

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%i in ("%SCRIPT_DIR%") do set "PROJECT_ROOT=%%~dpi"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Creating release archive for etx-tracer...
echo Project root: %PROJECT_ROOT%
echo Excluding: assets_testing/, lib/, tmp/

REM Create temporary directory for packaging
for /f "tokens=*" %%i in ('powershell -Command "[System.IO.Path]::GetTempPath() + [System.IO.Path]::GetRandomFileName()"') do set "TEMP_DIR=%%i"
set "RELEASE_DIR=%TEMP_DIR%\etx-tracer-release"

echo Using temporary directory: %TEMP_DIR%

REM Create release directory
mkdir "%RELEASE_DIR%" 2>nul

REM Copy only necessary directories from bin/
echo Copying bin/ contents (only including assets, fonts, spectrum)...
pushd "%PROJECT_ROOT%\bin"
for /d %%i in (*) do (
    if "%%i" equ "assets" (
        echo   Copying: %%i
        xcopy "%%i" "%RELEASE_DIR%\%%i\" /E /I /H /Y >nul
    ) else if "%%i" equ "fonts" (
        echo   Copying: %%i
        xcopy "%%i" "%RELEASE_DIR%\%%i\" /E /I /H /Y >nul
    ) else if "%%i" equ "spectrum" (
        echo   Copying: %%i
        xcopy "%%i" "%RELEASE_DIR%\%%i\" /E /I /H /Y >nul
    )
)
REM Copy necessary files (exe, dll, json)
for %%i in (*.exe *.dll *.json) do (
    echo   Copying: %%i
    copy "%%i" "%RELEASE_DIR%\" >nul
)
popd

REM Create zip of blender plugin
echo Creating blender plugin archive...
mkdir "%RELEASE_DIR%\blender" 2>nul
set "BLENDER_ZIP=%RELEASE_DIR%\blender\etx_tracer_exporter.zip"
pushd "%PROJECT_ROOT%\blender"
if exist "etx_tracer_exporter" (
    echo   Creating: blender/etx_tracer_exporter.zip
    powershell -Command "Compress-Archive -Path 'etx_tracer_exporter\*' -DestinationPath '%BLENDER_ZIP%' -Force"
    echo   Created: blender/etx_tracer_exporter.zip
) else (
    echo   Warning: Blender plugin directory not found
)
popd

REM Create final release archive
pushd "%TEMP_DIR%"
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyyMMdd-HHmmss'"') do set "TIMESTAMP=%%i"
set "ARCHIVE_NAME=etx-tracer-windows-%TIMESTAMP%.zip"

echo Creating final archive: %ARCHIVE_NAME%
powershell -Command "Compress-Archive -Path 'etx-tracer-release\*' -DestinationPath '%ARCHIVE_NAME%' -Force"

REM Move archive to project root
move "%ARCHIVE_NAME%" "%PROJECT_ROOT%\" >nul
set "FINAL_ARCHIVE=%PROJECT_ROOT%\%ARCHIVE_NAME%"

echo Archive created: %FINAL_ARCHIVE%

REM Cleanup
popd
rmdir /s /q "%TEMP_DIR%" 2>nul

echo.
echo Release archive creation complete!
echo Archive: %FINAL_ARCHIVE%
echo.
echo Contents:
powershell -Command "try { $zip = [System.IO.Compression.ZipFile]::OpenRead('%FINAL_ARCHIVE%'); $zip.Entries | Select-Object -First 20 | ForEach-Object { '{0,-10} {1}' -f $_.Length, $_.FullName }; $zip.Dispose() } catch { Write-Host 'Could not list archive contents' }"

echo.
echo To extract: powershell -Command "Expand-Archive -Path '%FINAL_ARCHIVE%' -DestinationPath '.\extracted'"

goto :eof