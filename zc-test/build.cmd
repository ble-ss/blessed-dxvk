@echo off
rem zero-copy-present seat: compiles zc_test.exe (BLESSED_ZERO_COPY verify
rem harness). compiles wait on the bench lock only (wait-bench), matching
rem present-cursed's pc-test/build.cmd -- GPU runs wait on quiet.sh
rem separately, see run.sh.
setlocal
set "HERE=%~dp0"
set "ROOT=E:\blessed_skyrim"

call "%ROOT%\tools\wait-bench.cmd" 2>nul

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64.bat failed & exit /b 1 )

pushd "%HERE%"
cl /nologo /std:c++17 /EHsc /W4 /O2 zc_test.cpp /link /out:zc_test.exe d3d11.lib dxgi.lib d3dcompiler.lib user32.lib gdi32.lib
if errorlevel 1 ( popd & echo zc_test.cpp build failed & exit /b 1 )
popd

echo built zc_test.exe.
endlocal
