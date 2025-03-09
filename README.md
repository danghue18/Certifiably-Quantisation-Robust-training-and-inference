# Certifiably Quantisation-Robust training and inference of Neural Networks
We tackle the problem of computing guarantees for the robustness of neural networks against quantisation of their inputs, parameters and activation values. In particular, we pose the problem of bounding the worst-case discrepancy between the original neural network and all possible quantised ones parametrised by a given maximum quantisation diameter $\epsilon > 0$ over a finite dataset. To achieve this, we first reformulate the problem  in terms of bilinear optimisation, which can be solved for provable bounds on the robustness guarantee. We then show how a quick scheme based on interval bound propagation can be developed and implemented during training so to allow for the learning of neural networks robust against a continuous family of quantisation techniques. 

## 1. How to Run the Program computing the worst-case discrepancy using bilinear optimization 
The below instruction is compute the worst-case discrepancy between the original neural network and all possible quantized ones over a finite test dataset by reformulating the problem in terms of bilinear optimization and using the Gurobi solver to solve it. Currently, we are experimenting with this method on 16 different models across 50 images from the MNIST test set.
 
Follow the steps below to run the experiments:

### Step 1: Add Gurobi License
1. Download a Gurobi license (we currently use an academic license https://www.gurobi.com/academia/academic-program-and-licenses/), rename as 'gurobi.lic" to your root directory (examples for Windows: C:\Users\hueda\gurobi.lic) because this is the default directory where Gurobi will search for the license.

### Step 2: Navigate to the Verification Folder
1. Move to the verification directory:
    ```bash
    cd verification
    ```

### Step 3: Run the Experiments
 1. Run the `run_experiments.sh` file:
    ```bash
    ./run_experiments.sh
    ```
    Note: Using run.experiments.bat for Windows. 
2. If the `run_experiments.sh` file does not have execution permission, add it using the following command:
    ```bash
    chmod +x run_experiments.sh
    ```

The program will parallelly execute 16 `.py` files, corresponding to 16 different experiments (16 different models).
To run 4/16 experiments, use: 

 ```bash
 ./sub_run1.sh
 ./sub_run2.sh
 ./sub_run3.sh
 ./sub_run4.sh
 ```

To run a single experiment, use: 
   ```bash
   ./file_name.py
   ```
The file_name values are defined in the run_experiments.sh file mentioned above.

## 2. How to Run the Program computing the worst-case discrepancy using interval bound propagation (IBP)
### Step 1: Navigate to the interval bound propagation directory
```bash
cd interval bound propagation
```
### Step 2: Run the `compute_worst_case_discrepancy.py` file:
```bash
./compute_worst_case_discrepancy.py
```
Note: To compute the worst-case discrepancy for a specific model, modify the architecture and its checkpoint as defined in compute_worst_case_discrepancy.py

## 3. How to train a robust model
```bash
cd interval bound propagation
./train_robust_model_MNIST
```

## 4. How to verify the robustness of a model against quantization using bilinear optimization
Do step 1 and step 2 above.
To evaluate the verified accuracy of robust models and the normal one with epsilon equivalent to the perturbation of 6 bits, 8 bits, 10 bits, 16 bits quantization.
With robust models, run the `run_verification.sh` file:
 ```bash
 ./run_verification.sh
 ```
To verify the normal models (standard training), use:
  ./sub_run_verification_3.sh

## 5. How to verify the robustness of a model against quantization using IBP
```bash
cd interval bound propagation
./compare_rob_acc.py
```

## 6. Empirical analysis of the effects of quantization on IBP-trained models - normal models - QAT-trained models - QAT-KURE-trained models. 
The training process for QAT-trained models and QAT-KURE-trained models is adapted from the GitHub repository: https://github.com/moranshkolnik/RobustQuantization.git
### 1. Compare the quantised accuracy in the CIFAR10 dataset
```bash
cd Robust_quantization-ref
```
To train a QAT-KURE model: 

```bash
./cnn_classifier_train_kurtosis.py -a mlp_cifar10 --custom_mlp --dataset cifar10  -q -bw 8 -ba 8 -ep 60 -exp temp --qtype lsq  --w-kurtosis --weight-name all --w-lambda-kurtosis 1.0 --w-kurtosis-target 1.8
```

To train a QAT model: 

```bash
 ./cnn_classifier_train.py -a mlp_cifar10 --custom_mlp --dataset cifar10  -q -bw 8 -ba 8 -ep 60 -exp temp --qtype lsq 
```
Compute the quantised accuracy: 

```bash
cd quantization
./extract_params_cifar10 (python extract_params_cifar10_9bits.py)
./determine_weight_scalars_cifar10.py 
./compute_quantization_error_cifar10.py ( python compute_qat_acc_with_PTQ.py)
```
### Compute the worst-case discrepancy across various quantisation precisions in the F-MNIST dataset

```bash
cd quantization
./extract_params_fmnist
./determine_weight_scalars_fminst.py 
./compute_quantization_error_fmist.py
```
