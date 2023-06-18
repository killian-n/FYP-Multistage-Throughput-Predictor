#!/bin/bash
current_directory=$(pwd)
find src -type d -exec cp project.env {} \;

cd src/data_transformation

# run_prefix=multivariate
# data_prefix=multivariate
# python create_datasets.py --prefix $data_prefix --include RSRQ SNR NRxRSRP CQI RSSI NRxRSRQ RSRP UL_bitrate State

# run_prefix=univariate
# data_prefix=univariate
# python create_datasets.py --prefix $data_prefix

# cd ../model_optimization

# python  optimize_model.py --model baseline --data_prefix multivariate

cd ../training_models
run_prefix=standardised
data_prefix=multivariate
python train_model.py --prefix $run_prefix --model "baseline" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "high" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "medium" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "low" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "classifier" --data_prefix $data_prefix

cd ../trained_models
python test_model.py --model_prefix $run_prefix --model "baseline" --data_prefix $data_prefix
python test_model.py --model_prefix $run_prefix --model "multiOne" --data_prefix $data_prefix
python test_model.py --model_prefix $run_prefix --model "multiAll" --data_prefix $data_prefix

cd ../training_models
run_prefix=standardised
data_prefix=univariate
python train_model.py --prefix $run_prefix --model "baseline" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "high" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "medium" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "low" --data_prefix $data_prefix
python train_model.py --prefix $run_prefix --model "classifier" --data_prefix $data_prefix

cd ../trained_models
python test_model.py --model_prefix $run_prefix --model "baseline" --data_prefix $data_prefix
python test_model.py --model_prefix $run_prefix --model "multiOne" --data_prefix $data_prefix
python test_model.py --model_prefix $run_prefix --model "multiAll" --data_prefix $data_prefix

