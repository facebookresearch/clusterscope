---
sidebar_position: 7
---

# get_tmp_dir()

`get_tmp_dir()` returns `/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}/`.

`get_tmp_dir()` in order of preference, returns:
- if Slurm Job, `/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}/` 
- `/tmp`

```python
import clusterscope
temp_dir = clusterscope.get_tmp_dir()
print(temp_dir)
# /scratch/slurm_tmpdir/<job-id>/
```

