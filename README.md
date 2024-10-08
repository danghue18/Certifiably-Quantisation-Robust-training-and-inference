# Certifiably Quantisation-Robust training and inference of Neural Networks
We address the problem of computing the worst-case discrepancy between the original neural network and the quantized one over a finite test dataset. To achieve this, we first reformulate the problem in terms of bilinear optimization and use the Gurobi solver to solve it. Currently, we are experimenting with this method on 16 different models across 50 images from the MNIST test set.

## How to Run the Program computing the worst-case discrepancy

Follow the steps below to run the experiments:

### Step 1: Add Gurobi License
1. Open the `licenses` folder and add the `gurobi_2.lic` file to your root directory (examples for Windows: C:\Users\hueda\gurobi.lic).
   - This is the default directory where Gurobi will search for the license.

### Step 2: Navigate to the Verification Folder
1. Move to the verification directory:
    ```bash
    cd verification
    ```

### Step 3: Run the Experiments
1. Modify ROOT in the .env file with your root path. 
2. Run the `run_experiments.sh` file:
    ```bash
    ./run_experiments.sh
    ```
    Note: Using run.experiments.bat for Windows. 
3. If the `run_experiments.sh` file does not have execution permission, add it using the following command:
    ```bash
    chmod +x run_experiments.sh
    ```

The program will sequentially execute 16 `.py` files, corresponding to 16 different experiments.
To run 4/16 experiments, use: 

 ```bash
 ./sub_run1.sh
 ./sub_run2.sh
 ./sub_run3.sh
 ./sub_run4.sh
 ```

### Step 4: Run Experiments in Parallel (Optional)
To optimize execution time, you can run multiple `.py` files in parallel using different terminals. This may be more efficient than using the `.sh` file:
   ```bash
   python file_name.py
   ```
The file_name values are defined in the run_experiments.sh file mentioned above.

## How to Run the Program verifying the robustness of a model against quantization 
Run the `run_verification.sh` file:
 ```bash
 ./run_experiments.sh
 ```
The program will sequentially execute 4 .py files to test the robust accuracy of robust models and the normal one with epsilon equivalent to the perturbation of 6 bits, 8 bits, 10 bits, 16 bits quantization.

To run 2/4 experiments, use: 

 ```bash
 ./sub_run_verification_1.sh
 ./sub_run_verification_2.sh

To verify the normal model, use:
  ./sub_run_verification_3.sh

