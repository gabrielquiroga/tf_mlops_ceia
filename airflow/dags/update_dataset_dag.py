from airflow.decorators import dag, task
from datetime import datetime, timedelta

default_args = {
    "owner": "gabi",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "dagrun_timeout": timedelta(minutes=25)
}

@dag(
    dag_id="update_dataset_dag",
    description="Descarga y actualiza el dataset en s3 diariamente",
    default_args=default_args,
    catchup=False,
    tags=["dataset", "s3", "update"],
    start_date=datetime(2025, 1, 1),
    schedule="@daily"
)
def update_dataset_dag():

    @task(task_id="download_and_upload_to_s3")
    def download_and_upload_to_s3():
        """
        Descarga el dataset desde la url configurada y lo sube al bucket
        S3 de MinIO.
        """
        import os
        import boto3
        import requests
        from io import BytesIO

        # URL del dataset
        dataset_url = "https://raw.githubusercontent.com/gabrielquiroga/ceia-amq1-tf/main/dataset/star_classification.csv"
        
        # Descarga
        print(f"Descargando datos desde {dataset_url}")

        response = requests.get(dataset_url)
        response.raise_for_status()

        # Cliente S3
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3', "http://minio:9000"),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', "minioadmin"),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', "minioadmin")
        )

        # Subida
        bucket = "datasets"
        key = "star_classification.csv"
        print(f"Subiendo el dataset a s3://{bucket}/{key}")

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=response.content,
            ContentType="text/csv"
        )

        print(f"Dataset actualizado exitosamente en S3")
        return f"s3://{bucket}/[{key}]"

    # Workflow simple
    download_and_upload_to_s3()

dag = update_dataset_dag()