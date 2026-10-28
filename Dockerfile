FROM public.ecr.aws/lambda/python:3.13
# Lambda's filesystem is read-only except /tmp
ENV HOME=/tmp \
    FASTEMBED_CACHE=/opt/fastembed
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt
# Bake the embedding model into the image so cold starts never download it
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed')"
COPY src/main.py src/data_loader.py src/vector_db.py src/custom_types.py ${LAMBDA_TASK_ROOT}/
CMD ["main.handler"]
