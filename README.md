# Certifiably Quantisation-Robust training and inference of Neural Networks
We address the problem of computing the worst-case discrepancy between the original neural network and the quantized one over a finite test dataset. To achieve this, we first reformulate the problem in terms of bilinear optimization and use the Gurobi solver to solve it. Currently, we are experimenting with this method on 16 different models across 50 images from the MNIST test set.

## How to Run the Program

Follow the steps below to run the experiments:

### Step 1: Add Gurobi License
1. Open the `licenses` folder and add the `gurobi.lic` file to your root directory (examples for both Windows: C:\Users\hueda\gurobi.lic).
   - This is the default directory where Gurobi will search for the license.

### Step 2: Navigate to the Verification Folder
1. Move to 'Verification' directory:
    ```bash
    cd verification
    ```

### Step 3: Run the Experiments
1. Run the `run_experiments.sh` file:
    ```bash
    ./run_experiments.sh
    ```

2. If the `run_experiments.sh` file does not have execution permission, add it using the following command:
    ```bash
    chmod +x run_experiments.sh
    ```

The program will sequentially execute 16 `.py` files, corresponding to 16 different experiments.

### Step 4: Run Experiments in Parallel (Optional)
To optimize execution time, you can run multiple `.py` files in parallel using different terminals. This may be more efficient than using the `.sh` file:
   ```bash
   python file_name.py