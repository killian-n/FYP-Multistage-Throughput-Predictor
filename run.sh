#!/bin/bash
current_directory=$(pwd)
find src -type d -exec cp project.env {} \;

conda activate tf

cd src/data_transformation

run_prefix=multivariate
data_prefix=multivariate
python create_datasets.py --prefix $data_prefix --include RSRQ SNR NRxRSRP CQI RSSI NRxRSRQ RSRP UL_bitrate State

run_prefix=univariate
data_prefix=univariate
python create_datasets.py --prefix $data_prefix

cd ../model_optimization

python  optimize_model.py --model baseline --data_prefix univariate

# cd ../training_models
# python.exe train_model.py --prefix $run_prefix --model "baseline" --data_prefix $data_prefix
# python.exe train_model.py --prefix $run_prefix --model "high" --data_prefix $data_prefix
# python.exe train_model.py --prefix $run_prefix --model "medium" --data_prefix $data_prefix
# python.exe train_model.py --prefix $run_prefix --model "low" --data_prefix $data_prefix
# python.exe train_model.py --prefix $run_prefix --model "classifier" --data_prefix $data_prefix

# cd ../trained_models
# python.exe test_model.py --model_prefix $run_prefix --model "baseline" --data_prefix $data_prefix
# python.exe test_model.py --model_prefix $run_prefix --model "multiOne" --data_prefix $data_prefix
# python.exe test_model.py --model_prefix $run_prefix --model "multiAll" --data_prefix $data_prefix

