#!/bin/bash

PROJECT_ID="mlops-deploy-497300"

gcloud builds submit \
--tag gcr.io/$PROJECT_ID/ataque-cardiaco-api

gcloud run deploy ataque-cardiaco-api \
--image gcr.io/$PROJECT_ID/ataque-cardiaco-api \
--platform managed \
--region us-central1 \
--allow-unauthenticated