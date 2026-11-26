@echo off
rem Build only the tiny standalone CUDA operator checker, NOT libuipc.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -shared -O2 -arch=sm_86 -o "%~dp0psq_operator_check.dll" "%~dp0psq_operator_check.cu"
exit /b %errorlevel%
