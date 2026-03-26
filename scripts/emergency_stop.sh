#!/bin/bash
echo "🚨 EMERGENCY STOP - Shutting down all resources"

# Set your details
PROJECT_ID="ai-innovation-486521"
REGION="europe-west4"  # or us-central1, wherever you deployed

# List and undeploy all endpoints
echo "Finding active endpoints..."
gcloud ai endpoints list --region=$REGION --project=$PROJECT_ID

echo ""
echo "Enter endpoint ID to stop (or 'all'):"
read ENDPOINT_ID

if [ "$ENDPOINT_ID" = "all" ]; then
    echo "Stopping all endpoints..."
    gcloud ai endpoints list --region=$REGION --project=$PROJECT_ID --format="value(name)" | while read endpoint; do
        echo "Undeploying $endpoint..."
        gcloud ai endpoints undeploy-model $endpoint --region=$REGION --project=$PROJECT_ID --deployed-model-id=all
    done
else
    echo "Undeploying endpoint $ENDPOINT_ID..."
    gcloud ai endpoints undeploy-model $ENDPOINT_ID --region=$REGION --project=$PROJECT_ID --deployed-model-id=all
fi

echo "✓ Stopped!"
