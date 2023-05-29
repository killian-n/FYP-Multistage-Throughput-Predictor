@echo off
set "SCRIPT_DIR=%~dp0"
REM REPLACE WITH YOUR PYTHON ENVIRONMENT
set python_environment="C:/Users/Killian/miniconda3/Scripts/activate"
set "PYTHONPATH=%PYTHONPATH%;%SCRIPT_DIR%"
for /r %SCRIPT_DIR%"\src" %%i in (.) do @copy %SCRIPT_DIR%"project.env" "%%i" > nul
call %python_environment% tf

cd src\data_transformation

set data_prefix="multivariate"
set run_prefix="multivariate"
python.exe create_datasets.py --prefix %data_prefix% --include RSRQ SNR NRxRSRP CQI RSSI NRxRSRQ RSRP UL_bitrate State

set data_prefix="univariate"
set run_prefix="univariate"
python.exe create_datasets.py --prefix %data_prefix%

cd ..\model_optimization
set data_prefix="multivariate"
python.exe optimize_model.py --model low --data_prefix %data_prefix%

@REM cd src\data_transformation
@REM python.exe create_datasets.py --prefix %data_prefix% --include RSRQ SNR NRxRSRP State NetworkMode
@REM cd ..\training_models
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix% --use_balanced True
@REM python.exe train_model.py --prefix %run_prefix% --model "high" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "medium" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "low" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "classifier" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "classifier" --data_prefix %data_prefix% --use_balanced True

@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiOne" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiAll" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix% --use_balanced True
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiOne" --data_prefix %data_prefix% --use_balanced True --prefix %run_prefix%_up
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiAll" --data_prefix %data_prefix% --use_balanced True --prefix %run_prefix%_up

@REM set run_prefix="univariate"
@REM set data_prefix="univariate"
@REM cd ..\data_transformation
@REM python.exe create_datasets.py --prefix %run_prefix%
@REM cd ..\training_models
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "high" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "medium" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "low" --data_prefix %data_prefix%
@REM python.exe train_model.py --prefix %run_prefix% --model "classifier" --data_prefix %data_prefix%

@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiOne" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiAll" --data_prefix %data_prefix%

@REM set run_prefix="H15H5"
@REM set data_prefix="H15H5"
@REM cd ..\data_transformation
@REM python.exe create_datasets.py --prefix %run_prefix% --history_window 15 --horizon_window 5
@REM cd ..\training_models
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%

@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%

@REM set run_prefix="H20H5"
@REM set data_prefix="H20H5"
@REM cd ..\data_transformation
@REM python.exe create_datasets.py --prefix %run_prefix% --history_window 20 --horizon_window 5
@REM cd ..\training_models
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%

@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%

@REM set run_prefix="H20H10"
@REM set data_prefix="H20H10"
@REM cd ..\data_transformation
@REM python.exe create_datasets.py --prefix %run_prefix% --history_window 20 --horizon_window 10
@REM cd ..\training_models
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%

@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%

@REM set run_prefix="constraint_1_5"
@REM set data_prefix="optimal"
@REM set float_value=15
@REM cd src\data_transformation
@REM @REM python.exe create_datasets.py --prefix %data_prefix% --include RSRQ SNR NRxRSRP State NetworkMode
@REM cd ..\training_models
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix% --size_constraint %float_value%
@REM @REM python.exe train_model.py --prefix %run_prefix% --model "high" --data_prefix %data_prefix% --size_constraint %float_value%
@REM @REM python.exe train_model.py --prefix %run_prefix% --model "medium" --data_prefix %data_prefix% --size_constraint %float_value%
@REM @REM python.exe train_model.py --prefix %run_prefix% --model "low" --data_prefix %data_prefix% --size_constraint %float_value%
@REM @REM python.exe train_model.py --prefix %run_prefix% --model "classifier" --data_prefix %data_prefix% --size_constraint %float_value%
@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM @REM python.exe test_model.py --model_prefix %run_prefix% --model "multiOne" --data_prefix %data_prefix%
@REM @REM python.exe test_model.py --model_prefix %run_prefix% --model "multiAll" --data_prefix %data_prefix%

cd ..\..
@REM set run_prefix="constraint_3"
@REM set data_prefix="optimal"

@REM cd src\data_transformation
@REM python.exe create_datasets.py --prefix %data_prefix% --include RSRQ SNR NRxRSRP State NetworkMode
@REM cd ..\training_models
@REM python.exe train_model.py --prefix %run_prefix% --model "baseline" --data_prefix %data_prefix% --size_constraint 3
@REM python.exe train_model.py --prefix %run_prefix% --model "high" --data_prefix %data_prefix% --size_constraint 3
@REM python.exe train_model.py --prefix %run_prefix% --model "medium" --data_prefix %data_prefix% --size_constraint 3
@REM python.exe train_model.py --prefix %run_prefix% --model "low" --data_prefix %data_prefix% --size_constraint 3
@REM python.exe train_model.py --prefix %run_prefix% --model "classifier" --data_prefix %data_prefix% --size_constraint 3
@REM cd ..\trained_models
@REM python.exe test_model.py --model_prefix %run_prefix% --model "baseline" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiOne" --data_prefix %data_prefix%
@REM python.exe test_model.py --model_prefix %run_prefix% --model "multiAll" --data_prefix %data_prefix%