@setlocal enableextensions enabledelayedexpansion
@echo off

set INPUT_REL_FILE=%1
set INPUT_FILE=%2
set FILE_SIZE=%3

set FORMAT=1
if not x%INPUT_REL_FILE:samplerBlueNoise=%==x%INPUT_REL_FILE% set FORMAT=0

if %FORMAT%==0 echo Skipping %INPUT_FILE%
if %FORMAT%==1 clang-format -verbose -style=file -i %INPUT_FILE%

endlocal
@echo on
