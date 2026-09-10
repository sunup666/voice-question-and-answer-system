@echo off

set var=
set path=

@REM self-define python path
set newpypath=%cd%\..\py310
set path=%path%;%newpypath%
set path=%path%;%newpypath%\Scripts

call pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

:go
set /p var=%cd% $ 
if "%var%"=="exit" goto end
%var%

goto go

:end