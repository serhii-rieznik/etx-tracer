@echo off
REM ###########################################################################################
REM #                                                                                         #
REM # this batch script should be called directly from root folder                            #
REM #                                                                                         #
REM ########################################################################################### 

set ROOT=%CD%
set FOLDERS=(sources)
set EXTENSIONS=(h, hpp, c, cpp, cxx, mm, glsl, hlsl, cu)
for %%f in %FOLDERS% do (
 for %%e in %EXTENSIONS% do ( 
  forfiles /s /p "%ROOT%\%%f" /m *.%%e /c "cmd /c %ROOT%\scripts\format-file.bat @relpath @path @fsize"
 )
)
@echo on

