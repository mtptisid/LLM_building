# 🧠 Distributed Training Guide with Horovod & DeepSpeed

> A comprehensive guide to setting up large-scale distributed training using Horovod and DeepSpeed for AI/ML workloads.

---

## 🧭 Table of Contents

- [📌 Why Distributed Training?](#why-distributed-training)
- [⚙️ Core Concepts](#core-concepts)
- [🚀 Horovod Setup](#horovod-setup)
  - [Prerequisites](#prerequisites)
  - [Docker + Kubernetes](#docker--kubernetes)
  - [Training Example](#training-example)
- [⚡ DeepSpeed Setup](#deepspeed-setup)
  - [Key Features](#key-features)
  - [Installation & Environment](#installation--environment)
  - [Training Example](#training-example-deepspeed)
- [🔁 Comparison: Horovod vs DeepSpeed](#comparison-horovod-vs-deepspeed)
- [📊 Monitoring Distributed Jobs](#monitoring-distributed-jobs)
- [📦 Deployment Tips](#deployment-tips)
- [📬 Support & Contact](#support--contact)

---

## 📌 Why Distributed Training?

Training large AI/ML models (e.g. LLMs, ResNet, GPT) can be time-consuming and require:

- Multiple GPUs or nodes
- Efficient communication between workers
- Parallelization of computation and memory usage

Distributed training allows faster training by:
- Splitting data across GPUs (Data Parallelism)
- Splitting model across devices (Model Parallelism)
- Overlapping communication with computation

---

## ⚙️ Core Concepts

- **Data Parallelism**: Each GPU gets a mini-batch and syncs gradients after backward pass.
- **Model Parallelism**: Layers are split across GPUs to train large models.
- **AllReduce**: Collective communication pattern for aggregating gradients.
- **Mixed Precision**: FP16 + FP32 for speed and reduced memory.

---

## 🚀 Horovod Setup

Horovod by Uber is built on MPI/NCCL for seamless data-parallel training.

### Prerequisites

- Python 3.8+
- OpenMPI or NCCL
- PyTorch/TensorFlow/Keras

### Docker + Kubernetes

```
FROM horovod/horovod:latest-py3.8-torch-cuda
RUN pip install tensorflow torch torchvision
```

Deploy on Kubernetes (example):

```
kubectl apply -f k8s/horovod-job.yaml
```

### Training Example

```
import horovod.torch as hvd
import torch

hvd.init()
torch.cuda.set_device(hvd.local_rank())
model = MyModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = hvd.DistributedOptimizer(optimizer)

# Wrap model
model = torch.nn.parallel.DistributedDataParallel(model)
```

Use MPI to launch:

```
horovodrun -np 8 -H localhost:4,host2:4 python train.py
```

---

## ⚡ DeepSpeed Setup

DeepSpeed by Microsoft is optimized for massive model training.

### Key Features

- ZeRO (Zero Redundancy Optimizer)
- 3D Parallelism (Data + Pipeline + Tensor)
- Offloading to CPU/NVMe
- Support for LLMs (GPT-NeoX, OPT, etc.)

### Installation & Environment

```
pip install deepspeed
```

Create `ds_config.json`:

```
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

### Training Example (DeepSpeed)

```
# train.py
import deepspeed
model = MyModel()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config_params=json.load(open('ds_config.json'))
)
for batch in data_loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

Launch:

```
deepspeed train.py --deepspeed --deepspeed_config ds_config.json
```

---

## 🔁 Comparison: Horovod vs DeepSpeed

| Feature              | Horovod                 | DeepSpeed                   |
|----------------------|--------------------------|-----------------------------|
| Best for             | Multi-GPU Data Parallel  | Massive Model Training      |
| Backend              | MPI, NCCL                | PyTorch Native              |
| Optimizer Support    | SGD, Adam                | ZeRO, Adam, Offload         |
| Mixed Precision      | Yes                      | Yes (NVIDIA Apex, native)   |
| Model Parallel       | Limited                  | Advanced 3D Parallelism     |
| Ease of Use          | Easy                     | Moderate (Powerful)         |

---

## 📊 Monitoring Distributed Jobs

- Use TensorBoard + Horovod Timeline
- Use Weights & Biases for DeepSpeed
- Use Prometheus/Grafana for GPU metrics

---

## 📦 Deployment Tips

- Use NCCL + EFA for AWS or GCP GPUs
- Prefer A100/H100 over V100s for speed
- Group GPUs on same node for performance
- Use Slurm or KubeRay for job orchestration

---

## 📬 Support & Contact

Need help setting this up or optimizing for your model?

📧 Email: msidrm455@gmail.com  
🌐 Portfolio: https://mtptisid.github.io

---

