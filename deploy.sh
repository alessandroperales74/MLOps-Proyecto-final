#!/bin/bash

PROJECT_ID="mlops-deploy-497300"

IMAGE_URI="us-central1-docker.pkg.dev/$PROJECT_ID/ataque-cardiaco-api/ataque-cardiaco-api"

gcloud builds submit \
--tag $IMAGE_URI

gcloud run deploy ataque-cardiaco-api \
--image $IMAGE_URI \
--platform managed \
--region us-central1 \
--allow-unauthenticated