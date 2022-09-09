variable "bucket_name" {
  default = "wojtek-ml-project"
}

variable "service-account" {
  type = string
  description = "Service Account associated with the whole function service"
}
