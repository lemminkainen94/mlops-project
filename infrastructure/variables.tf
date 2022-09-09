variable "gcp_region" {
  description = "Region in GCP"
  default = "eu-central2"
}

variable "gcp_project_id" {
  description = "Project ID in GCP"
}

variable "gcp_credentials" {
  default = "./terraform-service.json"
}

variable "prefix" {
  description = "Prefix for the services"
  default = "wojtel-ml-project"
}

variable "credentials" {
  default = "./terraform-service.json"
}
variable "service-account" {
}

variable "data_bucket_name" {
  default = "wojtek-ml-project"
}
