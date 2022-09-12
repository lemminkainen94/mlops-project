# Mlops Final Project - Target-Based Sentiment Analysis

Given a text and a target (e.g. Named Entity), predit the sentiment towards that entity.

## Target-Based Sentiment Analysis

In a classic sentiment analysis setting, we're given a set of documents (could be sentences or longer texts) and have to predict the sentiment (polarity) of each document. It is usually defined as a binary (0 - negative, 1 - positive) or ternary (-1 for negative, 0 for neutral, 1 for positive) classification.

In target-based sentiment analysis, we're given a set of documents and for each document - a target (usually a named entity) and have to predict the sentiment towards the target.

For instance, given a sentence "Everything was great, from food to service, but the Tiramisu was awful." The overall sentiment might be considered positive, the sentiment towards service positive as well, but the sentiment towards Tiramisu is clearly negative.

We'd like to train a model which is able to distinguish such nuances. This can be very useful for restaurants, hotels, etc... for analyzing which part of their services may need improvements, based on a significant amount of negative comments.

## Dataset

I've compiled a dataset based on the training data from [SemEval 2014](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval) and [SentiHood](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-sentihood) datasets. The dataset can be found in the data directory of the project. Since it's rather small, there is no problem keeping it in the repo.

data_prep.py can be used to split the dataset into current and future dataset (needed for monitoring later) and further divide the current data into train and test data. All files are then uploaded to a cloud storage.

The data is avialable in a tsv format (like csv but tab-separated). It contains 3 columns:
text: the text from which we want to predict the sentiment towards the target;
target: usually some named entity, but can be any noun.
sentiment: towards the target. Can be -1 for negative, 0 for neutral, 1 for positive.

## Metrics
Since the dataset is slightly unbalanced, with the class neutral being slightly underrepresented, I decided to use macro f1 to evaluate the performance of the models.

## Model
I preprocess the data with a CountVectorizer transformation, followed by Tfidf transformation, which is a popular data preprocessing method for text classification for non-deep models.
I decided to use SGDClassifier from scikit-learn. This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
I experiment with different losses:
- `log` loss gives logistic regression;
- `hinge` gives a linear SVM;
- `modified_huber` is another smooth loss;
- `perceptron` uses the perceptron algorithm.

## Training

1. Read the data, preprocess it using a CountVectorizer and TfidfTransformer.
2. Run a Hyperoptimzer for SGDClassifier hyperparameters, store results (Without the models) in the `tbsa-train` experiment. I did it for 100 iterations, but you can try a different number.
3. Get the best run from the previous experiment and retrain them and log the entire model and metrics with preprocessors in the `chosen-models-tbsa` experiment along with a version tag
4. Manually choose the best model and get its run id (for instance from the browser UI).
5. Go to train dir and run `python register_model.py <ID>` to register the model, measure its test metric and upload it, along with the preprocessors, to a GCS Bucket.

## Deployment

### Pre-requisites

1. Make sure you have the following tools installed: `terraform`, `pipenv`, `make`, `gcloud`. (Setup with `gcloud init`)
2. Create two service accounts with admin rights, one for Terraform and the other for running things (e.g. uploading and downloading from GCS).
3. Download JSON keys for each of the accounts. Put the Terraform's JSON key in `infrastructure` as `terraform-service.json` or whatever name you prefer. You can keep the other in the root of the project dir, or wherever/
4. Create a baseenv file (an example provided, feel free to modify), with the following definitions:
 + `GOOGLE_APPLICATION_CREDENTIALS`: The Absolute Path to the JSON key for the host account (Non-Terraform)
 + `GOOGLE_ACCOUNT_NAME`: email address of the host account
 + `PROJECT_ID`: ID of the GCP Project
5. Run `pipenv install`. If it doesn't work, try `python -m pipenv install`. And activate `[python -m ]pipenv shell`

### Deploy

Once the above pre-requisites are fullfilled, should be able to just run `make build`. This should prepare the datasets, initiate and apply Terraform infrastructure (buckets and the google function), initiate the envs and upload the model and the data to your bucket.
Note that this will already trigger the google function to run against the new data in the bucket and save the predictions, so you can see the function logs if it succeeded and if you have `predictions.tsv` file in the `data` directory of you bucket.
See the `Makefile` for other commands.

## Components

### Prefect

For scheduling, I use prefect and run it locally with `prefect orion start`. I also set up a String block `version-counter`. This value corresponds to the tag that will be set on the MLFlow runs. Keep the prefect running for training and monitoring.

### Training and Experiment Tracking

I'm using MLFlow for experiment tracking and Hyperopt for parameter tuning. In order to run training yourself, you can do the following:
1. Start mlflow UI with `mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root gs://your-bucket/artifacts --host 0.0.0.0`
2. Start prefect server with: `prefect orion start`
3. Start the prefect agent on your queue `prefect agent start --work-queue "ml"`
4. Run `./prefect_run.sh` This will deploy the trainig task and run it with the agent started above.
5. Get the run id of the best model, go to `training` and run `python register_model.py <run_id>`.

### Monitoring

I use Evidently AI for monitoring. Since it is difficult to measure a data-drift on Tfidf features (there are thousands of them), I only generate the classification report to compare how the deployed model performs on the new vs past data.

In order to generate the html report:
1. Start prefect server with: `prefect orion start`
2. Start the prefect agent on your queue `prefect agent start --work-queue "ml"`
3. run `python monitoring/monitoring.py && prefect deployment run batch_analyze/evidently-report`. This should generate an html file you can view in your browser.

### Testing

`pytest -m offline` for unit tests and `pytest -m online` for an integration test

### Pre-commit hooks

`.pre-commit-config.yaml` defines all hooks used in this project.
1. Some pre-defined hook, pylint and black are used for linting and formatting.
2. `isort` for sorting the libraries.
3. `pytest -m offline` runs unit tests.

### Continuous Integration

`.github/workflows/ci-test.yaml` defines some simple env setup and tests to run as some minimalisitc CI example.

## Destroying the Infrastructure:

To destroy the infrastructure, go to `infrastructure` and run `./terraform-destroy.sh`
