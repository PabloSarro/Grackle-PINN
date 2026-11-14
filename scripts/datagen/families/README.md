## 🚀 Study: Families of Trajectories

This folder examines the different set of possible behaviours that the problem shows depending on the initial conditions for Temperature ($T_0$), Neutral Hydrogen ($nHI_0$), and Ionized Hydrogen ($nHII_0$).


### 1. Initial Setup

```bash
pip install -r requirements.txt
```


### 2. Data Generation (Pre-requisite)
First, the whole box of possible initial conditions was grid-sampled, which is done in the program `ICs_grid.py`. Then, the corresponding Grackle Ground Truth files are generated with `generate_data.py`, which outputs a set of `output_{i}.dat` files. Later on, `find_families.py` performs an analysis on the families based on the system behaviour through `sklearn`, which generates a `Trajectory_Families.csv` file containing the classification for each output file. Finally `plot_families.py` shows a collection of trajectories for each family, to visualise the common behaviour for each of them in a set of `.png` images. This can be run in the following way:

**2.1. Create Initial Conditions:**
Run:
```bash
python ICs_grid.py
```

**2.2. Generate Ground Truth Simulations for all ICs:**
Run:
```bash
python generate_data.py ICs_grid.txt /scratch/izar/sarro/Grid_Analysis/Grid_Files # Or build SLURM job for better performance, if inside SCITAS.
```

**2.3. Model Training**
```bash
python find_families.py
```

**2.4. Result Analysis**
```bash
python plot_families.py
```

### 3. Visualisation of ICs per family (Optional)
Optionally, `map_ICs_families.py` shows the boundaries that the `sklearn` model predicted or used for each trajectory family:

```bash
python map_ICs_families.py
```

### 4. Generate a set of balanced ICs.
With `ICs_balanced.py`, we train a kNN model and randomly sample from the big box of temperatures and species densities, but this time ensuring that we sample 250 ICs from each family based on a kNN with $k=5$ (optionally, we can plot the balanced ICs against the previously grid-sampled ICs with `plot_ICs_balanced.py`).

```bash
python ICs_balanced.py
python plot_ICs_balanced.py # Optional
```



### 5. Dataset Creation and Split
With `split_ICs.py`, we divide the `ICs_balanced.txt` file (containing 1000 balanced ICs) into two separate `train.txt` (800 files, 200 of each family) and `valid.txt` (200 files, 50 of each family). With those, we build our training and validation stratified datasets using `generate_data.py`:

```bash
python split_ICs.py
python generate_data.py ICs_train.txt /scratch/izar/sarro/Balanced_Dataset/train
python generate_data.py ICs_valid.txt /scratch/izar/sarro/Balanced_Dataset/valid
```

---