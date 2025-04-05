# 🚀 Complete AI/ML Cluster Setup Guide with GPU/TPU Support

> Build a production-ready cluster for hosting AI/ML workloads or high-performance AI applications.

---

## 🧭 Table of Contents

- [📌 Objective](#objective)
- [🧠 Architecture Overview](#architecture-overview)
- [🛠️ Infrastructure Setup](#infrastructure-setup)
  - [GCP with GPU](#gcp-with-gpu)
  - [GCP with TPU](#gcp-with-tpu)
  - [On-Prem GPU Cluster](#on-prem-gpu-cluster)
- [🐳 Docker Environment for AI/ML](#docker-environment-for-aiml)
- [☸️ Kubernetes + Kubeflow Setup](#kubernetes--kubeflow-setup)
- [📦 Model Deployment (Triton/Custom)](#model-deployment-tritoncustom)
- [📊 Monitoring & Autoscaling](#monitoring--autoscaling)
- [🧪 Testing Setup](#testing-setup)
- [🔐 Security & Access Control](#security--access-control)
- [🧬 Multi-User AI IDE (JupyterHub/CodeServer)](#multi-user-ai-ide-jupyterhubcodeserver)
- [🌐 Ingress, DNS & SSL](#ingress-dns--ssl)
- [📁 Folder Structure](#folder-structure)
- [🚀 Sample Use Case: Deploying a Chatbot Model](#sample-use-case-deploying-a-chatbot-model)
- [📌 Coming Soon](#coming-soon)
- [📬 Support & Contact](#support--contact)

---

## 📌 Objective

To create a scalable, production-ready AI cluster that supports:

- GPU/TPU compute for training & inference
- ML pipelines and experiment tracking
- Multi-user Jupyter IDE
- High-availability inference APIs
- Real-time model monitoring

---

## 🧠 Architecture Overview

### Cloud + GPU + TPU

```
+------------------------------------+
| GCP GKE Cluster                    |
|  + Node Pool (A100 GPU)           |
|  + TPU Pod (v4-8, PyTorch/XLA)    |
+----------------+------------------+
                 |
    +------------v------------+
    |   Kubeflow Pipelines    |
    |  (TFX, Katib, MLFlow)   |
    +------------+------------+
                 |
    +------------v-------------+
    | JupyterHub / CodeServer  |
    +--------------------------+
                 |
    +------------v-----------+
    | Triton / FastAPI / GRPC |
    +------------------------+
```

---

## 🛠️ Infrastructure Setup

### GCP with GPU

```
gcloud services enable compute.googleapis.com container.googleapis.com

gcloud container clusters create ai-gpu-cluster \
  --zone=us-central1-a \
  --accelerator type=nvidia-tesla-a100,count=1 \
  --machine-type=n1-standard-8 \
  --num-nodes=3 \
  --enable-ip-alias \
  --scopes=cloud-platform
```

Install NVIDIA device plugin:

```
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
```

### GCP with TPU

```
gcloud services enable tpu.googleapis.com

gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central1-b \
  --accelerator-type=v4-8 \
  --version=tpu-vm-base
```

Inside TPU VM:

```
pip install torch_xla jax jaxlib tensorflow
```

### On-Prem GPU Cluster

- Install K8s with kubeadm
- Configure containerd and NVIDIA drivers
- Install NVIDIA K8s device plugin
- Optional: Configure Ceph or NFS for shared storage

---

## 🐳 Docker Environment for AI/ML

### Dockerfile (GPU/ML)

```
FROM nvidia/cuda:12.2.0-base-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip git curl
RUN pip3 install torch torchvision torchaudio tensorflow jupyterlab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
```

### Build & Run

```
docker build -t ml-gpu-env .
docker run --gpus all -p 8888:8888 ml-gpu-env
```

---

## ☸️ Kubernetes + Kubeflow Setup

Install kustomize and Kubeflow manifests:

```
curl -s https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh | bash

# Clone & deploy Kubeflow
https://github.com/kubeflow/manifests.git
cd manifests
while ! kustomize build example | kubectl apply -f -; do sleep 10; done
```

---

## 📦 Model Deployment (Triton/Custom)

### Triton Inference Server

```
kubectl apply -f k8s/triton-server.yaml
```

### Custom FastAPI Wrapper

```
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("model.pt")

@app.post("/predict")
def predict(input: dict):
    with torch.no_grad():
        output = model(torch.tensor(input["data"]))
    return {"result": output.tolist()}
```

---

## 📊 Monitoring & Autoscaling

- Prometheus + Grafana
- NVIDIA DCGM Exporter
- HPA + VPA for model autoscaling

```
kubectl apply -f dcgm-exporter.yaml
kubectl apply -f grafana-prometheus-stack.yaml
```

---

## 🧪 Testing Setup

### GPU Test

```
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### TPU Test

```
import torch_xla.core.xla_model as xm
print(xm.xla_device())
```

---

## 🔐 Security & Access Control

- Enable RBAC for Kubeflow
- Use GCP IAM roles
- Integrate Keycloak for multi-user login
- Enable Istio/NGINX auth + SSL

---

## 🧬 Multi-User AI IDE (JupyterHub/CodeServer)

### Deploy via Helm

```
helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
helm upgrade --install jhub jupyterhub/jupyterhub \
  --namespace jupyter --create-namespace \
  --values jupyter_config.yaml
```

Or deploy CodeServer with LDAP/AD login for enterprise users.

---

## 🌐 Ingress, DNS & SSL

- Use NGINX or Istio for ingress
- Integrate with external DNS (Cloudflare, GCP DNS)
- Auto-renew Let's Encrypt with cert-manager

```
kubectl apply -f ingress-nginx.yaml
kubectl apply -f cert-manager.yaml
```

---

## 📁 Folder Structure

```
.
├── Dockerfile.gpu-ml
├── k8s/
│   ├── triton-server.yaml
│   └── gpu-plugin.yaml
├── manifests/
│   └── kubeflow/
├── helm/
│   └── jupyter_config.yaml
└── terraform/
    └── gcp-cluster.tf
```

---

## 🚀 Sample Use Case: Deploying a Chatbot Model

1. Convert model to TorchScript or ONNX
2. Deploy to Triton with REST endpoint
3. Frontend app (React/Streamlit) calls backend
4. Backend uses Triton gRPC for prediction
5. Prometheus monitors latency & throughput

---

## 📌 Coming Soon

- [ ] Distributed Training (Horovod, DeepSpeed)
- [ ] Argo Workflows for CI/CD
- [ ] Advanced RBAC, Audit Logging

---

## 📬 Support & Contact

Need help setting this up? Want to host a GenAI application?

📧 Email: msidrm455@gmail.com  
🌐 Portfolio: https://mtptisid.github.io

---

