@echo off
python gurobi_verif_MNIST_6bits.py &
python gurobi_verif_MNIST_8bits.py &
python gurobi_verif_MNIST_10bits.py &
python gurobi_verif_MNIST_normal.py &


pause