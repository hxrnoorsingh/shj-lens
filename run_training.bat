@echo off
REM Setup and run training for SHJ project

echo ========================================
echo SHJ Project - Model Training Setup
echo ========================================
echo.

REM Activate the conda environment
echo Activating conda environment...
call C:\Users\ekonk\anaconda3\Scripts\activate.bat project2_env

REM Install requirements
echo.
echo Installing Python packages...
pip install numpy matplotlib seaborn jupyter pandas scipy

REM Run training
echo.
echo ========================================
echo Training models...
echo ========================================
cd models
python train.py

echo.
echo ========================================
echo Training complete!
echo ========================================
pause
