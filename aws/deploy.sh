#!/bin/bash
# Deploy to the USER'S OWN AWS account. No account ID, region, or key
# is hardcoded here. The user supplies all three.
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_ACCOUNT_ID=123456789012
#   export ECR_REPO=voice-booking-orchestrator
#   ./aws/deploy.sh
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION to your own region}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID to your own account}"
ECR_REPO="${ECR_REPO:-voice-booking-orchestrator}"
IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null
docker build -t "$IMAGE" .
docker push "$IMAGE"
echo "Image pushed: $IMAGE"
echo "Next, in YOUR AWS console, create an App Runner service from this image,"
echo "set port 8000, and add DEEPGRAM_API_KEY in App Runner secrets only if"
echo "you want a shared server-side default. Otherwise leave it empty for"
echo "bring-your-own-key mode. For production data, point DATABASE_URL at"
echo "your own RDS Postgres instance."
