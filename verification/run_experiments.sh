#!/bin/bash
python gurobi_worst_case_dis_MNIST6layers_512.py &
python gurobi_worst_case_dis_MNIST5layers_512.py &
python gurobi_worst_case_dis_MNIST4layers_512.py &
python gurobi_worst_case_dis_MNIST3layers_512.py &
python gurobi_worst_case_dis_MNIST6layers_256.py &
python gurobi_worst_case_dis_MNIST5layers_256.py &
python gurobi_worst_case_dis_MNIST4layers_256.py &
python gurobi_worst_case_dis_MNIST3layers_256.py &
python gurobi_worst_case_dis_MNIST3layers_64.py &
python gurobi_worst_case_dis_MNIST3layers_128.py &
python gurobi_worst_case_dis_MNIST4layers_64.py &
python gurobi_worst_case_dis_MNIST4layers_128.py &
python gurobi_worst_case_dis_MNIST5layers_64.py &
python gurobi_worst_case_dis_MNIST5layers_128.py &
python gurobi_worst_case_dis_MNIST6layers_64.py &
python gurobi_worst_case_dis_MNIST6layers_128.py &

wait