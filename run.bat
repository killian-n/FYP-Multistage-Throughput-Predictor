set "SCRIPT_DIR=%~dp0"
set python_environment="C:/Users/Killian/miniconda3/Scripts/activate"
set "PYTHONPATH=%PYTHONPATH%;%SCRIPT_DIR%"

call %python_environment% tf
set run_prefix="presentation"

cd src\data_transformation
REM python.exe create_datasets.py --prefix %run_prefix% --include RSRQ RSRP RSSI SNR CQI NRxRSRP NRxRSRQ
cd ..\training_models
python.exe train_model.py --prefix %run_prefix% --model "baseline"
python.exe train_model.py --prefix %run_prefix% --model "low"
python.exe train_model.py --prefix %run_prefix% --model "medium"
python.exe train_model.py --prefix %run_prefix% --model "high"
python.exe train_model.py --prefix %run_prefix% --model "classifier"

cd ..\trained_models
python.exe test_model.py --prefix %run_prefix% --model "baseline"
python.exe test_model.py --prefix %run_prefix% --model "multiOne"
python.exe test_model.py --prefix %run_prefix% --model "multiAll"
