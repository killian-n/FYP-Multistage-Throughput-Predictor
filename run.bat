@echo off
set "SCRIPT_DIR=%~dp0"
REM REPLACE WITH YOUR PYTHON ENVIRONMENT
set python_environment="C:/Users/knola/miniconda3/Scripts/activate"
set "PYTHONPATH=%PYTHONPATH%;%SCRIPT_DIR%"
for /r %SCRIPT_DIR%"\src" %%i in (.) do @copy %SCRIPT_DIR%"project.env" "%%i" > nul

call %python_environment% tf
set run_prefix="presentation"
set univariate_prefix="univariate"

cd src\data_transformation
@echo on
REM python.exe create_datasets.py --prefix %run_prefix% --include RSRQ RSRP RSSI SNR CQI NRxRSRP NRxRSRQ
@echo off
cd ..\training_models
@echo on
python.exe train_model.py --prefix %run_prefix% --model "baseline"
python.exe train_model.py --prefix %run_prefix% --model "low"
python.exe train_model.py --prefix %run_prefix% --model "medium"
python.exe train_model.py --prefix %run_prefix% --model "high"
python.exe train_model.py --prefix %run_prefix% --model "classifier"
@echo off
cd ..\trained_models
@echo on
python.exe test_model.py --prefix %run_prefix% --model "baseline"
python.exe test_model.py --prefix %run_prefix% --model "multiOne"
python.exe test_model.py --prefix %run_prefix% --model "multiAll"


REM THIS TRAINS AND TEST UNIVARIATE
@echo off
cd ..\data_transformation
@echo on
python.exe create_datasets.py --prefix %univariate_prefix% 
@echo off
cd ..\training_models
@echo on
python.exe train_model.py --prefix %univariate_prefix% --model "baseline"
python.exe train_model.py --prefix %univariate_prefix% --model "low"
python.exe train_model.py --prefix %univariate_prefix% --model "medium"
python.exe train_model.py --prefix %univariate_prefix% --model "high"
python.exe train_model.py --prefix %univariate_prefix% --model "classifier"
@echo off
cd ..\trained_models
@echo on
python.exe test_model.py --prefix %univariate_prefix% --model "baseline"
python.exe test_model.py --prefix %univariate_prefix% --model "multiOne"
python.exe test_model.py --prefix %univariate_prefix% --model "multiAll"