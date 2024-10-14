import numpy as np

def read_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Sử dụng hàm để đọc nội dung của tệp npy
file_path = 'opt_results/exp1_best_logs.npy'
data = read_npy_file(file_path)
print(data)
