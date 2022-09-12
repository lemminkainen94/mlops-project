resource null_resource "zip_function_files" {
  provisioner "local-exec" {
    command = "touch app.zip && rm app.zip && zip -urj app.zip ../app/main.py ../app/requirements.txt ../tbsa-host.json"
  }
}

resource "google_storage_bucket_object" "app_archive"{
  name = "app.zip"
  bucket = var.bucket_name
  source = "./app.zip"
  depends_on = [ null_resource.zip_function_files ]
}


resource "google_cloudfunctions_function" "app" {
  name = "mlops-project-tbsa"
  runtime = "python39"

  entry_point = "endpoint"

  source_archive_bucket = var.bucket_name
  source_archive_object = google_storage_bucket_object.app_archive.name

  event_trigger {
    event_type = "google.storage.object.finalize"
    resource = var.bucket_name
  }

  environment_variables = {
    PROJECT_ID = var.project_id
    BUCKET = var.bucket_name
  }
}
