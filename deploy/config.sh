# Shared defaults. Your real values go in deploy/config.local.sh (gitignored),
# which is loaded last and overrides anything here.
export AWS_REGION=us-east-2
export LAMBDA_NAME=""        # aws lambda list-functions --query "Functions[].FunctionName"
export ECR_REPO=""           # <account>.dkr.ecr.<region>.amazonaws.com/<repo>
export S3_BUCKET=""          # your uploads bucket
export INSTANCE_ID=""        # i-0123...
export DOMAIN=""             # e.g. carlos-rag.duckdns.org

[ -f deploy/config.local.sh ] && source deploy/config.local.sh

export AWS_DEFAULT_REGION="$AWS_REGION"
# Your current public IP, looked up at run time and never stored (used to restrict SSH)
export MY_IP="${MY_IP:-$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')}"
