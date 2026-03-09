import subprocess

__all__ = ['launch_slurm_job']


def modify_bash_file(file_path, old_string, new_string):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace the old string with the new string in each line
    modified_lines = [line.replace(old_string, new_string) for line in lines]

    # Write the modified contents back to the file
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

def launch_slurm_job(script_path, env, exp_name, dependencies=None):
    if dependencies is None:
        command = ["sbatch", script_path]
    else:
        command = ["sbatch", f"--dependency=afterok:{':'.join(dependencies)}", script_path]
    modify_bash_file(script_path, "expName", exp_name)
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    modify_bash_file(script_path, exp_name, "expName")

    if process.returncode == 0:
        job_id = extract_job_id(stdout)
        print(f'Successfully submitted SLURM job {exp_name} with ID {job_id}')
        return job_id
    else:
        print(f'Error submitting SLURM job {exp_name}: {stderr.decode("utf-8")}')
        return None

def extract_job_id(sbatch_output):
    output_lines = sbatch_output.decode("utf-8").split('\n')
    # The last line of the sbatch output contains the job ID
    job_id_line = output_lines[-2]
    job_id = job_id_line.split()[-1]
    return job_id


if __name__ == "__name__":
    pass
