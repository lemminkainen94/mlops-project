terraform {
  required_version = ">=1.0"
  backend "gcs" {
    bucket = "tbsa-terraform-state"
    prefix = "tfstate-stg"
    credentials = "terraform-service.json"
  }
  required_providers {
    google = {
      source = "hashicorp/google"
  }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = "europe-central2"
  zone    = "europe-central2-a"
}

module "bucket" {
  source = "./modules/gs"
  bucket_name = "${var.prefix}-${var.bucket_name}"
  service-account = var.service-account
}


# Write .env files
resource "local_file" "env_file" {

  content = <<EOT
PROJECT_ID=${var.gcp_project_id}
BUCKET=${var.prefix}-${var.bucket_name}
EOT

  filename = "../terraformenv"
}
resource null_resource "func_env_file" {
  provisioner "local-exec" {
    command = "cd ../ && cp terraformenv app/.env"
  }
  depends_on = [local_file.env_file]
}


module "app" {
  source = "./modules/app"
  prefix = var.prefix
  project_id = var.gcp_project_id
  bucket_name = module.bucket.bucket_name
  depends_on = [ module.bucket, null_resource.func_env_file ]
}
