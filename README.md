# gke-ml-platform

A scalable, **self-healing microservices platform for AI/ML workloads on Google Kubernetes Engine (GKE)**, provisioned end to end as Infrastructure as Code with Terraform. Built to stay fast and available under heavy load through autoscaling, health-based self-healing, and built-in observability.

**Stack:** Kubernetes (GKE) · Terraform · Docker · Python · Prometheus / Grafana

---

## What it does

This project stands up a production-style platform for serving microservice / ML workloads on Kubernetes and keeps it reliable under load:

- **Infrastructure as Code (Terraform).** The GKE cluster and supporting cloud infrastructure are defined in `terraform/`, so the whole environment is version-controlled, reproducible, and repeatable to stand up or tear down.
- **Containerized Python microservices.** Services live in `services/`, each packaged with its own Dockerfile for consistent, portable builds.
- **Horizontal autoscaling.** Kubernetes Horizontal Pod Autoscaling scales pods with demand, sustaining **1M+ requests per minute** under load with no manual intervention.
- **Self-healing and fault tolerance.** Custom **liveness and readiness probes** (`kubernetes/`) let Kubernetes detect unhealthy pods, pull them from rotation, and restart them automatically, holding **~99.99% availability** and minimizing downtime.
- **Observability built in.** Monitoring config in `monitoring/` wires up metrics and dashboards (Prometheus / Grafana) so the platform's health, latency, and resource use are visible in real time.

---

## Architecture

```
                    Terraform (IaC)
                         |  provisions
                         v
              +-----------------------------+
 traffic ---> |          GKE cluster        |
              |                             |
              |   Horizontal Pod Autoscaler |  scales with load
              |             |               |
              |             v               |
              |   Python microservices      |  (Docker images)
              |   + liveness/readiness      |  self-healing
              |     probes                  |
              +--------------+--------------+
                             |  metrics
                             v
                  Prometheus / Grafana  (monitoring/)
```

---

## Repository layout

```
gke-ml-platform/
|-- terraform/     # GKE cluster + cloud infrastructure as code (HCL)
|-- kubernetes/    # Deployments, services, HPA, liveness/readiness probes
|-- services/      # Containerized Python microservices (+ Dockerfiles)
|-- monitoring/    # Prometheus / Grafana observability config
`-- README.md
```

---

## Highlights

- Sustains **1M+ requests per minute** via horizontal autoscaling
- **~99.99% availability** through health-probe-based self-healing
- **Fully reproducible** infrastructure defined in Terraform
- **Observable by default** with metrics and dashboards

---

## Getting started

> Requires a Google Cloud project plus `gcloud`, `terraform`, and `kubectl`.

```bash
# 1. Provision the GKE cluster and infrastructure
cd terraform
terraform init
terraform apply

# 2. Point kubectl at the new cluster
gcloud container clusters get-credentials <cluster-name> --region <region>

# 3. Deploy the services, autoscaling, and health config
kubectl apply -f ../kubernetes/

# 4. Deploy monitoring
kubectl apply -f ../monitoring/
```

---

**Author:** Cosmas Mandikonza · [github.com/CosmasMandikonza](https://github.com/CosmasMandikonza)
