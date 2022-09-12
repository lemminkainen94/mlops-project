##
# mlops-proejct Target-Based Sentiment Analysis
#
# @file
# @version 0.1

data:
	python -m pipenv run python ./data_prep.py
terraform:
	cp baseenv infrastructure/.env
	cd infrastructure && terraform init
	cd infrastructure && ./terraform-apply.sh
dotenv:
	bash initiate-dotenvs.sh
build: data terraform dotenv
	bash upload.sh
plan-prod:
	cp baseenv infrastructure/.env
	cd infrastructure && ./terraform-plan-prod.sh
# end