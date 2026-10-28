FROM public.ecr.aws/lambda/python:3.13
# Lambda's filesystem is read-only except /tmp
ENV LLAMA_INDEX_CACHE_DIR=/tmp/llama_cache \
    TIKTOKEN_CACHE_DIR=/tmp/tiktoken \
    HOME=/tmp
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt
COPY src/ ${LAMBDA_TASK_ROOT}/
CMD ["main.handler"]
