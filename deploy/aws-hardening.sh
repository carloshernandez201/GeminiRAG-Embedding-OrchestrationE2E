#!/usr/bin/env bash
# One-time AWS lockdown. Safe to re-run. Run from the repo root: bash deploy/aws-hardening.sh
set -euo pipefail
source deploy/config.sh
: "${LAMBDA_NAME:?}" "${S3_BUCKET:?}" "${INSTANCE_ID:?}" "${MY_IP:?}"
step() { echo; echo "== $*"; }

step "S3: block all public access + expire leftover uploads after 1 day"
aws s3api put-public-access-block --bucket "$S3_BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-lifecycle-configuration --bucket "$S3_BUCKET" --lifecycle-configuration '{
  "Rules": [{"ID": "expire-visitor-uploads", "Status": "Enabled",
             "Filter": {"Prefix": "uploads/"}, "Expiration": {"Days": 1}}]}'

step "IAM: least-privilege S3 access for the Lambda role"
LAMBDA_ROLE=$(aws lambda get-function-configuration --function-name "$LAMBDA_NAME" --query Role --output text | awk -F/ '{print $NF}')
aws iam put-role-policy --role-name "$LAMBDA_ROLE" --policy-name rag-uploads-read-delete \
  --policy-document "$(sed "s/BUCKET_NAME/$S3_BUCKET/" deploy/iam-lambda-policy.json)"
echo "Other policies on $LAMBDA_ROLE (remove anything broader than logging, e.g. AmazonS3FullAccess):"
aws iam list-attached-role-policies --role-name "$LAMBDA_ROLE" --query 'AttachedPolicies[].PolicyName' --output text

step "IAM: instance role for EC2 (S3 upload only + SSM so SSH can be closed)"
if ! aws iam get-role --role-name rag-ec2-role >/dev/null 2>&1; then
  aws iam create-role --role-name rag-ec2-role --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam create-instance-profile --instance-profile-name rag-ec2-profile >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name rag-ec2-profile --role-name rag-ec2-role
  sleep 10
fi
aws iam put-role-policy --role-name rag-ec2-role --policy-name rag-uploads-put \
  --policy-document "$(sed "s/BUCKET_NAME/$S3_BUCKET/" deploy/iam-ec2-policy.json)"
aws iam attach-role-policy --role-name rag-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
CURRENT=$(aws ec2 describe-iam-instance-profile-associations --filters Name=instance-id,Values="$INSTANCE_ID" \
  --query 'IamInstanceProfileAssociations[0].IamInstanceProfile.Arn' --output text)
if [ "$CURRENT" = "None" ]; then
  aws ec2 associate-iam-instance-profile --instance-id "$INSTANCE_ID" --iam-instance-profile Name=rag-ec2-profile >/dev/null
else
  echo "Instance already has profile $CURRENT; attach the rag-uploads-put policy to that role instead if it isn't rag-ec2-role."
fi

step "EC2: require IMDSv2"
aws ec2 modify-instance-metadata-options --instance-id "$INSTANCE_ID" --http-tokens required --http-endpoint enabled >/dev/null

step "Security group: 80/443 open, 8501 closed, SSH only from $MY_IP"
SG=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)
for p in 80 443; do
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port $p --cidr 0.0.0.0/0 2>/dev/null || true
done
for p in 8501 8000 22; do
  aws ec2 revoke-security-group-ingress --group-id "$SG" --protocol tcp --port $p --cidr 0.0.0.0/0 2>/dev/null || true
done
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MY_IP/32" 2>/dev/null || true
aws ec2 describe-security-groups --group-ids "$SG" \
  --query 'SecurityGroups[0].IpPermissions[].{port:FromPort,from:IpRanges[].CidrIp}' --output table

step "Elastic IP (stable address for DuckDNS)"
EIP=$(aws ec2 describe-addresses --filters Name=instance-id,Values="$INSTANCE_ID" --query 'Addresses[0].PublicIp' --output text)
if [ "$EIP" = "None" ]; then
  ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
  aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC" >/dev/null
  EIP=$(aws ec2 describe-addresses --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)
fi
echo "Point your DuckDNS name at: $EIP   (https://www.duckdns.org)"
