#!/usr/bin/env bash

cat baseenv terraformenv > .env

cp .env monitoring/.env
cp .env train/.env
cp .env app/.env
cp app/.env app/tests/.env