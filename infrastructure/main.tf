terraform {
  required_version = ">=1.0"
  backend "gcs" {
    bucket = "tbsa-terraform-state"
    prefix = "tfstate-stg"
    credentials = "terraform-account.json"
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

module "data_bucket" {
  source = "./modules/gs"
  bucket_name = "${var.prefix}-${var.data_bucket_name}"
  service-account = var.service-account
}


# Write .env files
resource "local_file" "env_file" {

  content = <<EOT
PROJECT_ID=${var.gcp_project_id}
DATA_BUCKET=${var.prefix}-${var.data_bucket_name}
EOT

  filename = "../terraformenv"
}
resource null_resource "func_env_file" {
  provisioner "local-exec" {
    command = "cd ../ && cp terraformenv function/.env"
  }
  depends_on = [local_file.env_file]
}


module "function" {
  source = "./modules/function"
  prefix = var.prefix
  project_id = var.gcp_project_id
  data_bucket_name = module.data_bucket.bucket_name
  depends_on = [ module.data_bucket, null_resource.func_env_file ]
}
