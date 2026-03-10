#!/usr/bin/env bash
set -euo pipefail

image="predict-future-sales-byoc"

account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
region=${region:-us-east-1}

uri="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

aws ecr describe-repositories --repository-names "${image}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${image}" >/dev/null

aws ecr get-login-password --region "${region}" | \
  docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

docker build --network sagemaker -t "${image}:latest" -f container/Dockerfile .
docker tag "${image}:latest" "${uri}"
docker push "${uri}"

echo "IMAGE_URI=${uri}"