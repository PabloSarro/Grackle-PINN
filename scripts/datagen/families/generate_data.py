import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor

BINARY = os.environ.get("GRACKLE_DRIVER", "scripts/datagen/cooling-full-integration-100yr")

def run_simulation(task_id, t0, n0, x0, output_dir):
    """Executes a single instance of the Grackle binary."""
    # task_id is the 1-based line number, or the 4th column of the IC line when given
    output_filename = os.path.join(output_dir, task_id if str(task_id).endswith(".dat") else f"output_{task_id}.dat")
    
    # Assuming the binary is in the current working directory where you submit the job
    cmd = [
        BINARY,
        str(t0),
        str(n0),
        str(x0),
        output_filename
    ]
    
    # Execute the binary. 
    # stdout/stderr are routed to DEVNULL to prevent terminal clutter during parallel execution.
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=1800)
        return True, task_id
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False, task_id

def main():
    # Setup argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description="Run Grackle simulations in parallel.")
    parser.add_argument("param_file", type=str, help="Path to the input .txt file with initial conditions.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where .dat files will be saved.")
    
    args = parser.parse_args()
    param_file = args.param_file
    output_dir = args.output_dir
    
    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(param_file):
        print(f"[!] CRITICAL: Parameter file not found at {param_file}")
        return
    
    # 2. Read initial conditions
    with open(param_file, 'r') as f:
        # Strip newlines and ignore empty lines
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} parameter sets from {param_file}.")
    print(f"Starting parallel execution, saving to '{output_dir}'...")

    # 3. Execute in parallel
    successful_tasks = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as executor:
        futures = []
        
        # SLURM arrays use 1-based indexing, replicated here with start=1
        for task_id, line in enumerate(lines, start=1):
            fields = line.split()
            t0, n0, x0 = fields[:3]
            if len(fields) > 3:
                task_id = fields[3]
            futures.append(
                executor.submit(run_simulation, task_id, t0, n0, x0, output_dir)
            )
        
        # Monitor completion
        for future in futures:
            success, tid = future.result()
            if success:
                successful_tasks += 1
            else:
                print(f"Task {tid} failed during execution.")

    print(f"Execution complete. {successful_tasks}/{len(lines)} outputs generated in '{output_dir}'.")

if __name__ == "__main__":
    main()