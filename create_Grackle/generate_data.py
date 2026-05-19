import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

def run_simulation(task_id, t0, n0, x0, output_dir):
    """Executes a single instance of the Grackle binary."""
    output_filename = os.path.join(output_dir, f"output_{task_id}.dat")
    
    cmd = [
        "./cooling-full-integration-100yr",
        str(t0),
        str(n0),
        str(x0),
        output_filename
    ]
    
    # Execute the binary. 
    # stdout/stderr are routed to DEVNULL to prevent terminal clutter during parallel execution.
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True, task_id
    except subprocess.CalledProcessError as e:
        return False, task_id

def main():
    param_file = "params.txt"
    output_dir = "Grackle_100yr_3"
    
    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Read initial conditions
    with open(param_file, 'r') as f:
        # Strip newlines and ignore empty lines
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} parameter sets. Starting parallel execution...")

    # 3. Execute in parallel
    # ProcessPoolExecutor automatically defaults to the number of logical processors on your machine
    successful_tasks = 0
    with ProcessPoolExecutor() as executor:
        futures = []
        
        # SLURM arrays use 1-based indexing, replicated here with start=1
        for task_id, line in enumerate(lines, start=1):
            t0, n0, x0 = line.split()
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