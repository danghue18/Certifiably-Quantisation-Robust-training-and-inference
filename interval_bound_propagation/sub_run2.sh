#!/bin/bash
python robust_train_process_k_2.py --k 0.3 &
python robust_train_process_k_2.py --k 0.4 &
python robust_train_process_k_2.py --k 0.5 &
python robust_train_process_k_2.py --k 0.6 &
python robust_train_process_k_2.py --k 0.7 &
python robust_train_process_k_2.py --k 0.8 &


wait