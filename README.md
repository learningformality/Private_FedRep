# Private Federated Learning Algorithm Comparison

This code compares two private federated learning algorithms: Private FedRep and Priv-AltMin (from Jain et al., 2021). Both utilize user-level differential privacy to enforce privacy in a federated learning setting. Both algorithms use the same privacy settings ($\epsilon, \delta$) for a fair comparison.

The main script `run_experiment_vary_epsilon` runs this comparison. The code can take a while to run if you simulate a large number of users (say over 20,000). Results are subject to change each for each run. On average, Private FedRep does better than Priv-AltMin and the result graph of loss over \epsilon for the former is an approximate upper bound for the graph of the latter.

## Core Components

* **Synthetic Data Generation (`generate_synthetic_data`)**: Makes synthetic data (train, test, final) for each user. All user data (features $X$) is created using an underlying representation matrix $U$ and a local vector $v_i$. The label $y$ is $X U v_i$ plus Gaussian noise.
* **Privacy Utilities**:
    * `clip_L2`: Limits the L2 norm (length) of a vector or tensor.
    * `gaussian_noise_std`: Calculates how much Gaussian noise to add for privacy.
* **OLS Utilities (`solve_ols_v`)**: Finds the local vector $v$ for a user using Ordinary Least Squares (OLS). Solves $v = (XU)^\dagger y$.
* **APriv-AltMin Implementation**:
    * `dp_embedding_update`: Updates the shared $U$ with differential privacy. It collects special products ($x_{ij}v_i^T$) from users, clips them, adds noise, and then solves an OLS problem to get the new $U$.
    * `apriv_altmin_ols_dp`: Runs the APriv-AltMin algorithm. It switches between updating each user's $v_i$ (using OLS) and updating the shared $U$ (with privacy).
* **Private FedRep Implementation (`train_fedrep_dp`)**:
    * Starts with a global $U$.
    * Then, for a number of rounds:
        1.  Each user updates their local $v_i$ using the current $U$ (with OLS).
        2.  Each user calculates how $U$ should change (gradient $\nabla_U L_i$).
        3.  These changes are clipped then averaged over user count.
        4.  Gaussian noise is added to the average for user-level differential privacy.
        5.  The shared $U$ is updated with respect to this noisy mean.
        6.  $U$ if orthonormal-ized via QR decomposition.
* **Local DP Gradient Descent (`local_dp_gradient_descent`)**: An optional, basic method for comparison. Each user updates their own model using gradient descent. Gradients are clipped and noise is added 
       at each step for privacy.
* **Spectral Initialization (`model_init`)**: Calculates the starting $U$ using a traditional technique with guarantees supplied by the Davis-Kahan Sine Theorem.
* **Evaluation (`compute_test_mse`, `run_experiment_vary_epsilon`)**:
    * `compute_test_mse`: Checks how well the model performs by calculating the average Mean Squared Error (MSE) on test data.
    * `run_experiment_vary_epsilon`: The execution of the script.

## Dependencies

* `numpy`
* `torch`
* `matplotlib`
* `csv`

## Usage

To run the experiment, just run the Python script after dealing with dependencies via virtual environments or otherwise.
The `run_experiment_vary_epsilon()` function in the script handles everything. It runs a reasonable test case automatically with a non-private local training comparison. We use non-private FedRep as a baseline
for Private FedRep and Priv-AltMin since it performs nearly identically to non-private AltMin.

```bash
python your_script_name.py
