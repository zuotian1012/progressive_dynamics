@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cl /nologo /EHsc /std:c++17 /O2 "%~dp0check_trial_energy.cpp" /Fo:"%~dp0check_trial_energy.obj" /Fe:"%~dp0check_trial_energy.exe"
if errorlevel 1 exit /b 1
"%~dp0check_trial_energy.exe"
exit /b %errorlevel%
