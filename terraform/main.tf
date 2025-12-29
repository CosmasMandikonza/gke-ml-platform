# ============================================================================
# TERRAFORM CONFIGURATION
# ============================================================================
# This file defines all our Google Cloud infrastructure.
# ---------------------------------------------------------------------------
# TERRAFORM SETTINGS
# ---------------------------------------------------------------------------
terraform {
  # Minimum Terraform version required
  required_version = ">= 1.0.0"
  # Providers we'll use
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  # Backend configuration (where to store state)
  # For production, use a GCS bucket instead of local
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "terraform/state"
  # }
}
# ---------------------------------------------------------------------------
# GOOGLE PROVIDER CONFIGURATION
# ---------------------------------------------------------------------------
provider "google" {
  project = var.project_id
  region  = var.region
}
# ---------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------
variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}
variable "region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-central1"
}
variable "zone" {
  description = "Google Cloud zone"
  type        = string
  default     = "us-central1-a"
}
variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "ml-platform-cluster"
}
variable "node_count" {
  description = "Number of nodes in the default node pool"
  type        = number
  default     = 2
}
variable "machine_type" {
  description = "Machine type for nodes"
  type        = string
  default     = "e2-medium"  # Cost-effective for learning
}
# ---------------------------------------------------------------------------
# VPC NETWORK
# ---------------------------------------------------------------------------
# A Virtual Private Cloud network isolates our resources
resource "google_compute_network" "vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false  # We'll create our own subnet
  description = "VPC network for ML platform"
}
# Subnet for our GKE cluster
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/16"  # IP range for this subnet
  region        = var.region
  network       = google_compute_network.vpc.id
  # Secondary IP ranges for GKE pods and services
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}
# ---------------------------------------------------------------------------
# GKE CLUSTER
# ---------------------------------------------------------------------------
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone
  # We'll manage nodes separately
  remove_default_node_pool = true
  initial_node_count       = 1
  # Network configuration
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
  # IP allocation for pods and services
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  # Enable Workload Identity (secure way for pods to access GCP)
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  # Addons
  addons_config {
    # Horizontal Pod Autoscaler
    horizontal_pod_autoscaling {
      disabled = false
    }
    # HTTP Load Balancing (for Ingress)
    http_load_balancing {
      disabled = false
    }
  }
  # Logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"
}
# ---------------------------------------------------------------------------
# NODE POOL
# ---------------------------------------------------------------------------
# Node pool contains the actual VMs that run our containers
resource "google_container_node_pool" "primary_nodes" {
  name       = "${var.cluster_name}-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  # Autoscaling configuration
  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }
  # Initial number of nodes
  initial_node_count = var.node_count
  # Node configuration
  node_config {
    machine_type = var.machine_type
    # Disk configuration
    disk_size_gb = 50
    disk_type    = "pd-standard"
    # OAuth scopes (what GCP APIs nodes can access)
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    # Labels for node selection
    labels = {
      env = "production"
    }
    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
  }
    }
  # Management settings
  management {
    auto_repair  = true   # Automatically repair unhealthy nodes
    auto_upgrade = true   # Automatically upgrade nodes
  }
}
# ---------------------------------------------------------------------------
# ARTIFACT REGISTRY
# ---------------------------------------------------------------------------
# Where we store our Docker images
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "ml-platform-images"
  description   = "Docker images for ML platform"
  format        = "DOCKER"
}
# ---------------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------------
# Values we'll need after Terraform creates resources
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}
output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}
output "region" {
  description = "Region"
  value       = var.region
}
output "zone" {
  description = "Zone"
  value       = var.zone
}
output "artifact_registry" {
  description = "Docker image registry URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/ml-platform-images"
}
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --zone ${var.zone} --project ${var.project_id}"
}