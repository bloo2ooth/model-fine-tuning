#!/bin/bash
# Run this in background to auto-stop endpoint after 5 hours

ENDPOINT_ID=$1
REGION=${2:-"europe-west4"}
HOURS=${3:-5}

echo "Will auto-stop endpoint $ENDPOINT_ID after $HOURS hours"

sleep $((HOURS * 3600))

echo "⏰ Time limit reached - stopping endpoint..."
gcloud ai endpoints undeploy-model $ENDPOINT_ID \
    --region=$REGION \
    --deployed-model-id=all

echo "✓ Endpoint stopped"
