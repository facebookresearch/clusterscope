---
sidebar_position: 6
---

# local_node_gpu_generation_and_count()

`local_node_gpu_generation_and_count()` returns a single or a list of GPUInfo instances:

```python
import clusterscope
gpu_info = clusterscope.local_node_gpu_generation_and_count()
print(gpu_info)
# GPUInfo(gpu_gen='H100', gpu_count=8, vendor='nvidia', partition=None)
```

