## 🚀 Execution Guide: Grackle-PINN Emulation

This project implements a **Physics-Informed Neural Network (PINN)** to emulate the Grackle chemistry and cooling library. The model solves stiff coupled ODEs for Temperature ($T$), Neutral Hydrogen ($HI$), and Ionized Hydrogen ($HII$).


### 1. Initial Setup

```bash
pip install -r requirements.txt
mkdir -p logs GrackleData
```


### 2. Data Generation (Pre-requisite)
Before training the neural network, you must generate the Ground Truth (GT) dataset using the Grackle library. To do so:

**2.1. Create Initial Conditions:**
Run:
```bash
python params.py
```

**2.2. Generate Ground Truth Simulations for all ICs:**
Run:
```bash
sbatch generate_data.sh
```


### 3. Model Training
Training is designed to run on a GPU-enabled cluster using SLURM. 

| Script | Purpose |
| :--- | :--- |
| `data_utils.py` | Handles $\log_{10}$ space conversion and unified IC-target normalization. |
| `model.py` | Defines the MLP/ResNet architecture with the **Initial Condition (IC) Anchor**. |
| `physics.py` | Implements the Analytical Grackle rates and Physics Loss residuals. |
| `train.py` | Main optimization loop using Adam and Gradient Clipping. |

**To launch on the Izar Cluster:**
```bash
sbatch train.job
```

### 4. Monitoring Progress
While the job is running, you can monitor the training dynamics through the generated SLURM logs:

* **Check status:** `squeue -u <your_gaspar_id>`
* **Live logs:** `tail -f train_<job_id>.log`
* **Checkpoints:** The model automatically saves `scaling_params_PINN.pth` (normalization boundaries) and intermediate checkpoints every 5 epochs.

### 5. Evaluation and Visualization
Once the model is trained, use the visualization script to compare the PINN predictions against the Grackle Ground Truth on a logarithmic scale.

```bash
python visualise.py
```
*This will generate `pinn_performance.png`, showcasing the model’s ability to capture non-linear chemical and thermal dynamics.*

---