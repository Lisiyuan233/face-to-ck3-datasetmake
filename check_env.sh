#!/usr/bin/env bash
cd /mnt/d/workspace/face-to-ck3-datasetmake
source .venv-linux/bin/activate
which python
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "devs", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
    print("mem(GB):", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
PY
