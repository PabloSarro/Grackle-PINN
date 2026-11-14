#!/bin/bash
#SBATCH --job-name=GenData
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --account=master
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=generate_data.log

# Load modules
module load gcc cmake git hdf5/1.12.2

# Activate your environment
cd /home/sarro/SemProj/Pablo/Grackle-PINN/create_Grackle/
source ../EnvPablo/bin/activate

# Execute your generation script
python generate_data.py