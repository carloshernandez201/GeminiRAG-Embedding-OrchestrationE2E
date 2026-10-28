#!/usr/bin/env bash
# Rebuild the Lambda image, push it to ECR, and harden the function config.
# Run from the repo root: bash deploy/lambda.sh
set -euo pipefail
source deploy/config.sh
: "${LAMBDA_NAME:?set in deploy/config.sh}" "${ECR_REPO:?set in deploy/config.sh}"

ARCH=$(aws lambda get-function-configuration --function-name "$LAMBDA_NAME" --query 'Architectures[0]' --output text)
PLATFORM=linux/amd64; [ "$ARCH" = "arm64" ] && PLATFORM=linux/arm64
TAG=$(git rev-parse --short HEAD)

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${ECR_REPO%%/*}"
docker build --platform "$PLATFORM" --provenance=false -t "$ECR_REPO:$TAG" .
docker push "$ECR_REPO:$TAG"

aws lambda update-function-code --function-name "$LAMBDA_NAME" --image-uri "$ECR_REPO:$TAG" >/dev/null
aws lambda wait function-updated --function-name "$LAMBDA_NAME"
# More memory = more CPU for embedding; capped concurrency so nobody can hammer it
aws lambda update-function-configuration --function-name "$LAMBDA_NAME" --memory-size 2048 --timeout 120 >/dev/null
aws lambda put-function-concurrency --function-name "$LAMBDA_NAME" --reserved-concurrent-executions 2 >/dev/null
echo "Deployed $ECR_REPO:$TAG to $LAMBDA_NAME"
echo "Now re-sync the app in Inngest: https://app.inngest.com (Apps -> rag_app -> Resync)"
