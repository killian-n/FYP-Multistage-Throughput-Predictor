@echo off
set "SCRIPT_DIR=%~dp0"
REM REPLACE WITH YOUR PYTHON ENVIRONMENT
set python_environment="C:/Users/Killian/miniconda3/Scripts/activate"
set "PYTHONPATH=%PYTHONPATH%;%SCRIPT_DIR%"
for /r %SCRIPT_DIR%"\src" %%i in (.) do @copy %SCRIPT_DIR%"project.env" "%%i" > nul
call %python_environment% tf

cd src\data_transformation

@REM set data_prefix="multivariate"
@REM set run_prefix="multivariate"
@REM python.exe create_datasets.py --prefix %data_prefix% --include RSRQ SNR NRxRSRP CQI RSSI NRxRSRQ RSRP UL_bitrate State

@REM set data_prefix="univariate"
@REM set run_prefix="univariate"
@REM python.exe create_datasets.py --prefix %data_prefix%

cd ..\model_optimization
set data_prefix="multivariate"
@REM python.exe -Wignore optimize_model.py --model baseline --data_prefix %data_prefix%
@REM python.exe -Wignore optimize_model.py --model classifier --data_prefix %data_prefix%
python.exe -Wignore optimize_model.py --model high --data_prefix %data_prefix%

@REM set data_prefix="multivariate"
@REM python.exe -Wignore optimize_model.py --model classifier --data_prefix %data_prefix%
@REM python.exe -Wignore optimize_model.py --model baseline --data_prefix %data_prefix%
@REM python.exe -Wignore optimize_model.py --model low --data_prefix %data_prefix%
@REM python.exe -Wignore optimize_model.py --model medium --data_prefix %data_prefix%

@REM cd ..\training_models
@REM set data_prefix="univariate"
@REM set run_prefix="univariate"
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "high" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "medium" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "low" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "classifier" --data_prefix %data_prefix%

@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiOne" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiAll" --data_prefix %data_prefix%