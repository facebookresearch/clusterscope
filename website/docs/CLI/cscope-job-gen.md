---
sidebar_position: 6
---

# cscope job-gen

:::tip

CLI version of [`cscope job-gen` command](../py_lib/job-gen-task-slurm)

:::

`cscope job-gen` generates Slurm arguments for [GPU Jobs](#cscope-job-gen-for-gpu-jobs) and [CPU Jobs](#cscope-job-gen-for-cpu-jobs). The user pass a number of Resources (GPUs or CPUs) and a Slurm partition and `job-gen` calculates the proportionate amount of CPUs and Memory to allocate given what's available in that partition. It also supports outputs to slurm shapes like sbatch (slurm directives), the slurm CLI, and [submitit](https://github.com/facebookincubator/submitit/). 

Sample command asking for half GPUs (4 out of 8) in a node and slurm_cli formatted output:

```
$ cscope job-gen task slurm --partition=h100 --gpus-per-task=4 --format=slurm_cli
--cpus-per-task=96 --mem=999G --ntasks-per-node=1 --partition=h100 --gpus-per-task=4
```

For all the options:

```shell
$ cscope job-gen task slurm --help
Usage: cscope job-gen task slurm [OPTIONS]

  Generate job requirements for a task of a Slurm job based on GPU or CPU per
  task requirements.

Options:
  --partition TEXT                Partition to query  [required]
  --tasks-per-node INTEGER        Number of tasks per node to request
  --nodes INTEGER                 Number nodes to request
  --format [json|sbatch|slurm_directives|slurm_cli|submitit]
                                  Format to output the job requirements in
  GPU or CPU Job Request: [mutually_exclusive, required]
                                  Only one of --gpus-per-task or --cpus-per-
                                  task can be specified. For GPU requests, use
                                  --gpus-per-task and cpus-per-task will be
                                  generated automatically. For CPU requests,
                                  use --cpus-per-task.
    --gpus-per-task INTEGER RANGE
                                  Number of GPUs per task to request  [x>=1]
    --cpus-per-task INTEGER RANGE
                                  Number of CPUs per task to request  [x>=1]
  --help                          Show this message and exit.
```

## cscope job-gen for GPU Jobs

Asking for 8 out of 8 GPUs, allocates all the Memory and CPU:
```shell
$ cscope job-gen task slurm --partition=h100 --gpus-per-task=8
{
  "cpus_per_task": 192,
  "memory": "1999G",
  "tasks_per_node": 1,
  "slurm_partition": "h100",
  "nodes": 1,
  "gpus_per_task": 8,
  "mem_gb": 1999
}
```

Asking for half of the GPUs, allocates half the Memory and CPU:
```shell
$ cscope job-gen task slurm --partition=h100 --gpus-per-task=4
{
  "cpus_per_task": 96,
  "memory": "999G",
  "tasks_per_node": 1,
  "slurm_partition": "h100",
  "nodes": 1,
  "gpus_per_task": 4,
  "mem_gb": 999
}
```


## cscope job-gen for CPU Jobs

Asking for all CPUs, allocates all the Memory:
```shell
$ cscope job-gen task slurm --partition=h100 --cpus-per-task=192
{
  "cpus_per_task": 192,
  "memory": "1999G",
  "tasks_per_node": 1,
  "slurm_partition": "h100",
  "nodes": 1,
  "mem_gb": 1999
}
```

Asking for half of  the CPUs, allocates half the Memory:
```shell
$ cscope job-gen task slurm --partition=h100 --cpus-per-task=96
{
  "cpus_per_task": 96,
  "memory": "999G",
  "tasks_per_node": 1,
  "slurm_partition": "h100",
  "nodes": 1,
  "mem_gb": 999
}
```

## Output Formats

### slurm_cli

```shell
$ cscope job-gen task slurm --partition=h100 --gpus-per-task=4 --format=slurm_cli
--cpus-per-task=96 --mem=999G --ntasks-per-node=1 --partition=h100 --gpus-per-task=4
```

### sbatch, slurm_directives

```shell
$ cscope job-gen task slurm --partition=h100 --gpus-per-task=4 --format=slurm_directives
#SBATCH --cpus-per-task=96
#SBATCH --mem=999G
#SBATCH --ntasks-per-node=1
#SBATCH --partition=h100
#SBATCH --gpus-per-task=4
```

### submitit

```shell
$ cscope job-gen task slurm --partition=h100 --gpus-per-task=4 --format=submitit
{
  "cpus_per_task": 96,
  "mem_gb": 999,
  "slurm_partition": "h100",
  "tasks_per_node": 1,
  "gpus_per_task": 4
}
```

### json

```shell
$ cscope job-gen task slurm --partition=h100 --gpus-per-task=4 --format=json
{
  "cpus_per_task": 96,
  "memory": "999G",
  "tasks_per_node": 1,
  "slurm_partition": "h100",
  "nodes": 1,
  "gpus_per_task": 4,
  "mem_gb": 999
}
```

## Slurm Job Launcher Example

Check out the example below on how to use cscope to launch jobs, observe how the user only specifies GPUs and other requirements like memory and CPU are derived for the user:

```bash
#!/bin/bash
SLURM_ACCOUNT=<insert-your-slurm-account>
PARTITION=<insert-your-slurm-partition>
NUM_GPUS=2
QOS=<insert-your-slurm-QOS>

CSCOPE_CMD=(cscope job-gen task slurm --gpus-per-task $NUM_GPUS --partition "$PARTITION" --nodes=1 --format slurm_cli)
# Read generated line into an array
if read -r -a GEN_ARGS < <("${CSCOPE_CMD[@]}"); then
  :
else
  echo "Error: cscope command failed." >&2
  exit 1
fi

jobid=$(sbatch --parsable \
  "${GEN_ARGS[@]}" \
    --job-name=test_cscope_num_gpu_${NUM_GPUS} \
    --account="$SLURM_ACCOUNT" \
    --qos=${QOS} \
    --time=0:10:00 \
    --output=logs/%x-%A_%a.out \
    --array=1-8 <<'EOF'
#!/bin/bash
echo "Hostname: $(hostname)"
EOF
)
echo "Submitted as $jobid"
```