#!/bin/bash

python gurobi_worst_case_dis_MNIST3layers_64.py &
python gurobi_worst_case_dis_MNIST3layers_128.py &
python gurobi_worst_case_dis_MNIST3layers_256.py &
python gurobi_worst_case_dis_MNIST3layers_512.py &

wait