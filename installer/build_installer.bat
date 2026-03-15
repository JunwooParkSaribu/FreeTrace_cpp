@echo off
REM Made by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
REM Build script for FreeTrace C++ Windows installer
REM
REM FULLY AUTOMATIC -- downloads all dependencies if missing.
REM Prerequisites: Visual Studio 2022 (C++), CMake, Python, CUDA 12.x
REM Everything else (cuDNN, ONNX Runtime, vcpkg, Inno Setup) is auto-downloaded.
REM
REM Usage: just double-click, or run from command prompt:
REM   build_installer.bat

call :main
echo.
pause
exit /b %ERRORLEVEL%

:main
setlocal enabledelayedexpansion

pushd "%~dp0.."
set "ROOT=!CD!"
popd
set "DEPS=%~dp0deps"
set "STAGING=!ROOT!\installer_staging"
set "BUILD_DIR=!ROOT!\build_win"

REM --- Version config (change these to update dependencies) ---
set "ORT_VERSION=1.24.3"
set "CUDNN_VERSION=9.2.1.18"
set "CUDNN_MAJOR=9"

echo ============================================================
echo FreeTrace C++ Windows Installer Build
echo ============================================================
echo.

REM ============================================================
REM  Pre-check: verify required tools are installed
REM ============================================================
echo Checking prerequisites ...
echo.
set "PREREQ_MISSING=0"

REM --- 1. CMake ---
where cmake >nul 2>&1
if not errorlevel 1 (
    echo   [OK] CMake
) else (
    echo   [MISSING] 1. CMake 3.18+
    echo             Download: https://cmake.org/download/
    echo             Check "Add CMake to the system PATH" during install
    echo             Also requires: Visual Studio 2022 with "Desktop development with C++"
    echo             Download VS: https://visualstudio.microsoft.com/downloads/
    set "PREREQ_MISSING=1"
)

REM --- 2. Python ---
where python >nul 2>&1
if not errorlevel 1 (
    echo   [OK] Python
) else (
    echo   [MISSING] 2. Python 3.10+
    echo             Download: https://www.python.org/downloads/
    echo             Check "Add python.exe to PATH" during install
    set "PREREQ_MISSING=1"
)

REM --- 3. Git (needed to clone vcpkg if missing) ---
where git >nul 2>&1
if not errorlevel 1 (
    echo   [OK] Git
) else (
    echo   [MISSING] 3. Git
    echo             Download: https://git-scm.com/download/win
    set "PREREQ_MISSING=1"
)

REM --- 4. CUDA Toolkit 12.x ---
set "CUDA_DIR="
for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*") do (
    if exist "%%D\bin\cudart64_12.dll" set "CUDA_DIR=%%D"
)
if defined CUDA_DIR (
    echo   [OK] CUDA 12.x -- !CUDA_DIR!
) else (
    echo   [MISSING] 4. NVIDIA CUDA Toolkit 12.x
    echo             Download: https://developer.nvidia.com/cuda-downloads
    echo             Requires an NVIDIA GPU and up-to-date GPU driver
    set "PREREQ_MISSING=1"
)

echo.
if not "!PREREQ_MISSING!"=="0" (
    echo ============================================================
    echo  Please install the missing prerequisites listed above,
    echo  then run this script again.
    echo  Install them in the numbered order shown.
    echo ============================================================
    exit /b 1
)
echo All prerequisites found.
echo.

REM --- Create deps directory ---
if not exist "!DEPS!" mkdir "!DEPS!"

REM ============================================================
REM  Step 0: Find / download remaining dependencies (automatic)
REM ============================================================

REM --- 0a: Find or download cuDNN ---
echo [0a] Looking for cuDNN !CUDNN_MAJOR!.x ...
set "CUDNN_DIR="
REM Check if cuDNN is in CUDA dir
if exist "!CUDA_DIR!\bin\cudnn64_!CUDNN_MAJOR!.dll" (
    set "CUDNN_DIR=!CUDA_DIR!"
    echo     Found in CUDA dir: !CUDA_DIR!
    goto :cudnn_done
)
REM Check deps folder
for /d %%D in ("!DEPS!\cudnn-*") do (
    if exist "%%D\bin\cudnn64_!CUDNN_MAJOR!.dll" (
        set "CUDNN_DIR=%%D"
        echo     Found: %%D
    )
)
if defined CUDNN_DIR goto :cudnn_done
REM Download cuDNN
echo     Not found -- downloading cuDNN !CUDNN_VERSION! for CUDA 12 ...
set "CUDNN_ZIP=!DEPS!\cudnn.zip"
set "CUDNN_URL=https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-!CUDNN_VERSION!_cuda12-archive.zip"
echo     URL: !CUDNN_URL!
echo     This may take several minutes (cuDNN is ~700 MB) ...
curl.exe -L --progress-bar -o "!CUDNN_ZIP!" "!CUDNN_URL!"
if errorlevel 1 (
    echo ERROR: Failed to download cuDNN.
    echo        Download manually from https://developer.nvidia.com/cudnn
    echo        Extract to !DEPS!\
    exit /b 1
)
echo     Extracting ...
tar -xf "!CUDNN_ZIP!" -C "!DEPS!"
del "!CUDNN_ZIP!"
for /d %%D in ("!DEPS!\cudnn-*") do (
    if exist "%%D\bin\cudnn64_!CUDNN_MAJOR!.dll" set "CUDNN_DIR=%%D"
)
if not defined CUDNN_DIR (
    echo ERROR: cuDNN extraction failed -- cudnn64_!CUDNN_MAJOR!.dll not found.
    exit /b 1
)
echo     Extracted to: !CUDNN_DIR!
:cudnn_done

REM --- 0b: Find or download ONNX Runtime GPU ---
echo [0b] Looking for ONNX Runtime GPU !ORT_VERSION! ...
set "ORT_DIR="
REM Check deps folder
for /d %%D in ("!DEPS!\onnxruntime-win-x64-gpu*") do (
    if exist "%%D\lib\onnxruntime.dll" (
        set "ORT_DIR=%%D"
        echo     Found: %%D
    )
)
if defined ORT_DIR goto :ort_done
REM Check common locations
if exist "C:\onnxruntime-win-x64-gpu-!ORT_VERSION!\lib\onnxruntime.dll" (
    set "ORT_DIR=C:\onnxruntime-win-x64-gpu-!ORT_VERSION!"
    echo     Found: !ORT_DIR!
    goto :ort_done
)
REM Download ONNX Runtime
echo     Not found -- downloading ONNX Runtime GPU !ORT_VERSION! ...
set "ORT_ZIP=!DEPS!\onnxruntime.zip"
set "ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v!ORT_VERSION!/onnxruntime-win-x64-gpu-!ORT_VERSION!.zip"
echo     URL: !ORT_URL!
curl.exe -L --progress-bar -o "!ORT_ZIP!" "!ORT_URL!"
if errorlevel 1 (
    echo ERROR: Failed to download ONNX Runtime.
    echo        Download manually from https://github.com/microsoft/onnxruntime/releases
    exit /b 1
)
echo     Extracting ...
tar -xf "!ORT_ZIP!" -C "!DEPS!"
del "!ORT_ZIP!"
for /d %%D in ("!DEPS!\onnxruntime-win-x64-gpu*") do (
    if exist "%%D\lib\onnxruntime.dll" set "ORT_DIR=%%D"
)
if not defined ORT_DIR (
    echo ERROR: ONNX Runtime extraction failed.
    exit /b 1
)
echo     Extracted to: !ORT_DIR!
:ort_done

REM --- 0c: Find or setup vcpkg ---
echo [0c] Looking for vcpkg ...
set "VCPKG_ROOT="
if exist "!DEPS!\vcpkg\vcpkg.exe" (
    set "VCPKG_ROOT=!DEPS!\vcpkg"
    echo     Found: !DEPS!\vcpkg
    goto :vcpkg_libs
)
if exist "C:\vcpkg\vcpkg.exe" (
    set "VCPKG_ROOT=C:\vcpkg"
    echo     Found: C:\vcpkg
    goto :vcpkg_libs
)
REM Clone and bootstrap vcpkg
echo     Not found -- cloning vcpkg ...
git clone https://github.com/microsoft/vcpkg.git "!DEPS!\vcpkg"
if errorlevel 1 (echo ERROR: Failed to clone vcpkg && exit /b 1)
echo     Bootstrapping vcpkg ...
cmd /c ""!DEPS!\vcpkg\bootstrap-vcpkg.bat" -disableMetrics"
if errorlevel 1 (echo ERROR: vcpkg bootstrap failed && exit /b 1)
set "VCPKG_ROOT=!DEPS!\vcpkg"
echo     Installed: !VCPKG_ROOT!

:vcpkg_libs
set "VCPKG_INSTALLED=!VCPKG_ROOT!\installed\x64-windows"
REM Install tiff and libpng if not already present
if not exist "!VCPKG_INSTALLED!\bin\tiff.dll" (
    echo     Installing tiff and libpng via vcpkg ...
    "!VCPKG_ROOT!\vcpkg.exe" install tiff:x64-windows libpng:x64-windows
    if errorlevel 1 (echo ERROR: vcpkg install failed && exit /b 1)
) else (
    echo     tiff and libpng already installed
)

REM --- 0d: Find or download Inno Setup ---
echo [0d] Looking for Inno Setup ...
set "ISCC_CMD="
where iscc >nul 2>&1 && (
    set "ISCC_CMD=iscc"
    echo     Found: iscc in PATH
    goto :iscc_done
)
if exist "!DEPS!\innosetup\ISCC.exe" (
    set "ISCC_CMD=!DEPS!\innosetup\ISCC.exe"
    echo     Found: !DEPS!\innosetup\ISCC.exe
    goto :iscc_done
)
REM Check default install location
set "INNO_DEFAULT=!ProgramFiles(x86)!\Inno Setup 6\ISCC.exe"
if exist "!INNO_DEFAULT!" (
    set "ISCC_CMD=!INNO_DEFAULT!"
    echo     Found: !INNO_DEFAULT!
    goto :iscc_done
)
REM Download Inno Setup
echo     Not found -- downloading Inno Setup 6 ...
set "INNO_EXE=!DEPS!\innosetup_setup.exe"
set "INNO_URL=https://jrsoftware.org/download.php/is.exe"
curl.exe -L --progress-bar -o "!INNO_EXE!" "!INNO_URL!"
if errorlevel 1 (
    echo WARNING: Failed to download Inno Setup. Installer step will be skipped.
    goto :iscc_done
)
echo     Installing Inno Setup silently to !DEPS!\innosetup ...
cmd /c ""!INNO_EXE!" /VERYSILENT /SUPPRESSMSGBOXES /DIR="!DEPS!\innosetup" /CURRENTUSER /NORESTART"
if errorlevel 1 (
    echo WARNING: Inno Setup install failed. Installer step will be skipped.
) else (
    set "ISCC_CMD=!DEPS!\innosetup\ISCC.exe"
    echo     Installed: !DEPS!\innosetup
)
del "!INNO_EXE!" >nul 2>&1
:iscc_done

echo.
echo ============================================================
echo Dependencies ready:
echo   CUDA:         !CUDA_DIR!
echo   cuDNN:        !CUDNN_DIR!
echo   ONNX Runtime: !ORT_DIR!
echo   vcpkg:        !VCPKG_ROOT!
echo   Inno Setup:   !ISCC_CMD!
echo   Staging:      !STAGING!
echo ============================================================
echo.

REM ============================================================
REM  Step 1: Build freetrace.exe
REM ============================================================
echo === Step 1: Building freetrace.exe ===
if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cmake -S "!ROOT!" -B "!BUILD_DIR!" -G "Visual Studio 17 2022" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DUSE_CUDA=ON ^
    -DUSE_ONNXRUNTIME=ON ^
    -DONNXRUNTIME_DIR="!ORT_DIR!" ^
    -DCMAKE_TOOLCHAIN_FILE="!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake"
if errorlevel 1 (echo ERROR: CMake configure failed && exit /b 1)

cmake --build "!BUILD_DIR!" --config Release --parallel
if errorlevel 1 (echo ERROR: Build failed && exit /b 1)

REM ============================================================
REM  Step 2: Build FreeTrace_GUI.exe with PyInstaller
REM ============================================================
echo.
echo === Step 2: Building FreeTrace_GUI.exe ===
python -m pip install pyinstaller PyQt6 --quiet
python -m PyInstaller "!ROOT!\gui.spec" --noconfirm --clean --distpath "!ROOT!\dist" --workpath "!ROOT!\build_pyinstaller"
if errorlevel 1 (echo ERROR: PyInstaller failed && exit /b 1)

REM ============================================================
REM  Step 3: Stage all files
REM ============================================================
echo.
echo === Step 3: Staging files ===
if exist "!STAGING!" rmdir /s /q "!STAGING!"
mkdir "!STAGING!"
mkdir "!STAGING!\models"

REM --- Main executables ---
copy "!BUILD_DIR!\Release\freetrace.exe" "!STAGING!\" >nul 2>&1
if not exist "!STAGING!\freetrace.exe" copy "!BUILD_DIR!\freetrace.exe" "!STAGING!\" >nul
copy "!ROOT!\dist\FreeTrace_GUI.exe" "!STAGING!\" >nul

REM --- Model files ---
copy "!ROOT!\models\*.onnx" "!STAGING!\models\" >nul
copy "!ROOT!\models\*.bin" "!STAGING!\models\" >nul

REM --- Icon ---
mkdir "!STAGING!\icon"
copy "!ROOT!\icon\freetrace_icon.png" "!STAGING!\icon\" >nul 2>&1

REM --- ONNX Runtime DLLs ---
copy "!ORT_DIR!\lib\onnxruntime.dll" "!STAGING!\" >nul
copy "!ORT_DIR!\lib\onnxruntime_providers_cuda.dll" "!STAGING!\" >nul 2>&1
copy "!ORT_DIR!\lib\onnxruntime_providers_shared.dll" "!STAGING!\" >nul 2>&1

REM --- CUDA runtime DLLs ---
copy "!CUDA_DIR!\bin\cudart64_12.dll" "!STAGING!\" >nul
copy "!CUDA_DIR!\bin\cublas64_12.dll" "!STAGING!\" >nul
copy "!CUDA_DIR!\bin\cublasLt64_12.dll" "!STAGING!\" >nul
copy "!CUDA_DIR!\bin\cufft64_11.dll" "!STAGING!\" >nul 2>&1
copy "!CUDA_DIR!\bin\curand64_10.dll" "!STAGING!\" >nul 2>&1
copy "!CUDA_DIR!\bin\cusolver64_11.dll" "!STAGING!\" >nul 2>&1
copy "!CUDA_DIR!\bin\cusparse64_12.dll" "!STAGING!\" >nul 2>&1
copy "!CUDA_DIR!\bin\nvJitLink_120_0.dll" "!STAGING!\" >nul 2>&1
copy "!CUDA_DIR!\bin\nvrtc64_120_0.dll" "!STAGING!\" >nul 2>&1

REM --- cuDNN 9.x DLLs ---
copy "!CUDNN_DIR!\bin\cudnn64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul
copy "!CUDNN_DIR!\bin\cudnn_cnn64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul 2>&1
copy "!CUDNN_DIR!\bin\cudnn_ops64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul 2>&1
copy "!CUDNN_DIR!\bin\cudnn_graph64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul 2>&1
copy "!CUDNN_DIR!\bin\cudnn_engines_precompiled64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul 2>&1
copy "!CUDNN_DIR!\bin\cudnn_engines_runtime_compiled64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul 2>&1
copy "!CUDNN_DIR!\bin\cudnn_heuristic64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul 2>&1
copy "!CUDNN_DIR!\bin\cudnn_adv64_!CUDNN_MAJOR!.dll" "!STAGING!\" >nul 2>&1

REM --- zlibwapi.dll (required by cuDNN 9.x for heuristic engine) ---
copy "!CUDNN_DIR!\bin\zlibwapi.dll" "!STAGING!\" >nul 2>&1
copy "!CUDA_DIR!\bin\zlibwapi.dll" "!STAGING!\" >nul 2>&1
if not exist "!STAGING!\zlibwapi.dll" (
    echo     Downloading zlibwapi.dll for cuDNN ...
    curl.exe -L --progress-bar -o "!DEPS!\zlib123dllx64.zip" "https://www.winimage.com/zLibDll/zlib123dllx64.zip"
    if not errorlevel 1 (
        tar -xf "!DEPS!\zlib123dllx64.zip" -C "!DEPS!" dll_x64/zlibwapi.dll 2>nul
        copy "!DEPS!\dll_x64\zlibwapi.dll" "!STAGING!\" >nul 2>&1
        del "!DEPS!\zlib123dllx64.zip" >nul 2>&1
    )
)

REM --- libtiff + libpng + zlib (from vcpkg) ---
copy "!VCPKG_INSTALLED!\bin\tiff.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\tiffxx.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\libpng16.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\png16.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\zlib1.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\jpeg62.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\turbojpeg.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\liblzma.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\libwebp.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\libsharpyuv.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\Lerc.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\libdeflate.dll" "!STAGING!\" >nul 2>&1
copy "!VCPKG_INSTALLED!\bin\zstd.dll" "!STAGING!\" >nul 2>&1

REM --- Visual C++ Runtime ---
copy "!SystemRoot!\System32\vcruntime140.dll" "!STAGING!\" >nul 2>&1
copy "!SystemRoot!\System32\vcruntime140_1.dll" "!STAGING!\" >nul 2>&1
copy "!SystemRoot!\System32\msvcp140.dll" "!STAGING!\" >nul 2>&1
copy "!SystemRoot!\System32\concrt140.dll" "!STAGING!\" >nul 2>&1

echo.
echo Staged files:
dir /b "!STAGING!"
echo.
echo Models:
dir /b "!STAGING!\models"

REM ============================================================
REM  Step 4: Build installer
REM ============================================================
echo.
echo === Step 4: Building installer ===
if not defined ISCC_CMD (
    echo Inno Setup not available -- skipping installer build.
    echo Install from https://jrsoftware.org/isinfo.php
    echo Then run: iscc installer\freetrace_installer.iss
) else (
    "!ISCC_CMD!" "!ROOT!\installer\freetrace_installer.iss"
    if errorlevel 1 (echo ERROR: Inno Setup failed && exit /b 1)
    echo.
    echo Installer created: installer\FreeTrace_1.6.0.4_win64_setup.exe
)

echo.
echo ============================================================
echo Build complete!
echo ============================================================
endlocal
exit /b 0
