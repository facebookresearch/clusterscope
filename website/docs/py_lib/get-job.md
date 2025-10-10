---
sidebar_position: 3
---

# get_job()

`get_job()` gives data about the current running Job:

```python
import clusterscope
job = clusterscope.get_job()
```

Which then enables all of the methods below:

## get_cpus()

If inside a Slurm Job, returns the number of CPUs allocated on the Node according to Slurm ([SLURM_CPUS_ON_NODE](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_CPUS_ON_NODE)), if not, returns number of CPUs available on the local node. 

```python
import clusterscope
job = clusterscope.get_job()
cpus_count = job.get_cpus()
print(gpus_count, type(job_id))
# 80 <class 'int'>
```

## get_gpus()


If inside a Slurm Job, returns the number of GPUs allocated on the Node according to Slurm ([SLURM_GPUS_ON_NODE](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_GPUS_ON_NODE)), if not, returns number of GPUs available on the local node. 

```python
import clusterscope
job = clusterscope.get_job()
gpus_count = job.get_gpus()
print(gpus_count, type(job_id))
# 2 <class 'int'>
```

## get_job_id()

If inside a Slurm Job, returns the Job ID, if not, returns 0.

```python
import clusterscope
job = clusterscope.get_job()
job_id = job.get_job_id()
print(job_id, type(job_id))
# <job-id> <class 'int'>
```

## get_job_name()

If inside a Slurm Job, returns the Job ID, if not, returns 0.

```python
import clusterscope
job = clusterscope.get_job()
job_name = job.get_job_name()
print(job_name, type(job_name))
# <job-name> <class 'str'>
```

## get_global_rank()

Method to get the Global Rank of the current process. In order of preference, this returns:
- `RANK` env var
- if Slurm Job, `SLURM_PROCID` env var
- 0

```python
import clusterscope
job = clusterscope.get_job()
global_rank = job.get_global_rank()
print(global_rank, type(global_rank))
# 0 <class 'int'>
```

## get_local_rank()

Method to get the Local Rank of the current process. In order of preference, this returns:
- `LOCAL_RANK` env var
- if Slurm Job, `SLURM_LOCALID` env var
- 0

```python
import clusterscope
job = clusterscope.get_job()
local_rank = job.get_local_rank()
print(local_rank, type(local_rank))
# 0 <class 'int'>
```

## get_world_size()

Method to get the Local Rank of the current process. In order of preference, this returns:
- `WORLD_SIZE` env var
- if Slurm Job, `SLURM_NTASKS` env var
- 1

```python
import clusterscope
job = clusterscope.get_job()
world_size = job.get_world_size()
print(world_size, type(world_size))
# 10 <class 'int'>
```

## get_is_rank_zero()

Returns a boolean that indicates whether [`get_global_rank()`](#get_global_rank)

```python
import clusterscope
job = clusterscope.get_job()
is_rank_zero = job.get_is_rank_zero()
print(is_rank_zero, type(is_rank_zero))
# True <class 'bool'>
```

## get_master_port()

Method to get the Rendezvous Master Port. In order of preference, this returns:
- `MASTER_PORT` env var
- Random int from (20_000, 60_000), seeded at `SLURM_JOB_ID` or -1

```python
import clusterscope
job = clusterscope.get_job()
master_port = job.get_master_port()
print(master_port, type(master_port))
# 20_000 <class 'int'>
```

## get_master_addr()

Method to get the Rendezvous Master Address. In order of preference, this returns:
- `MASTER_ADDR` env var
- if Slurm Job, first node from: `scontrol show hostnames os.environ["SLURM_JOB_NODELIST"]`
- 127.0.0.1

```python
import clusterscope
job = clusterscope.get_job()
master_addr = job.get_master_addr()
print(master_addr, type(master_addr))
# 127.0.0.1 <class 'str'>
```

## set_torch_distributed_env_from_slurm()

Method to set torch distributed vars from Slurm vars. This assign values as below:

```
Torch Distributed, Slurm Var

WORLD_SIZE, SLURM_NTASKS
RANK, SLURM_PROCID
LOCAL_WORLD_SIZE, SLURM_NTASKS_PER_NODE
LOCAL_RANK, SLURM_LOCALID
MASTER_ADDR, get_master_addr()
MASTER_PORT get_master_port()
CUDA_VISIBLE_DEVICES, SLURM_LOCALID
```