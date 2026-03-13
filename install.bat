@echo off
REM ComfyUI-RayNodes Face Analysis Dependencies Installer
REM This script installs dependencies without overwriting CUDA-enabled PyTorch

echo Installing ComfyUI-RayNodes face analysis dependencies...
echo.

REM Install common dependencies
echo [1/2] Installing common dependencies...
pip install numpy pillow requests onnx onnxruntime opencv-python

REM Install packages that depend on torch without triggering torch reinstall
echo.
echo [2/2] Installing facenet-pytorch and emotiefflib (without deps)...
pip install facenet-pytorch emotiefflib --no-deps

echo.
echo ========================================
echo Installation complete!
echo.
echo If you want GPU acceleration for ONNX, run:
echo   pip install onnxruntime-gpu --upgrade
echo ========================================
pause
