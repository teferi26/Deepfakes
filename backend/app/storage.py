import hashlib
import uuid
from io import BytesIO
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from .config import settings


def get_s3_client():
    """Crea cliente S3 configurado para MinIO o AWS."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        use_ssl=settings.s3_secure,
    )


def ensure_bucket_exists(client=None):
    """Crea el bucket si no existe."""
    client = client or get_s3_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError:
        client.create_bucket(Bucket=settings.s3_bucket)


def compute_file_hash(file: BinaryIO) -> str:
    """Calcula SHA-256 del archivo."""
    sha = hashlib.sha256()
    for chunk in iter(lambda: file.read(8192), b""):
        sha.update(chunk)
    file.seek(0)
    return sha.hexdigest()


def upload_file(file: BinaryIO, filename: str, content_type: str) -> dict:
    """
    Sube archivo a S3/MinIO.
    Retorna dict con object_key, hash, size.
    """
    client = get_s3_client()
    ensure_bucket_exists(client)

    file_hash = compute_file_hash(file)
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    object_key = f"{uuid.uuid4().hex}.{ext}" if ext else uuid.uuid4().hex

    # Leer contenido para obtener tamaño
    file.seek(0)
    data = file.read()
    size = len(data)
    file.seek(0)

    client.upload_fileobj(
        BytesIO(data),
        settings.s3_bucket,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )

    return {
        "object_key": object_key,
        "bucket": settings.s3_bucket,
        "hash": file_hash,
        "size": size,
        "content_type": content_type,
    }


def download_file(object_key: str) -> bytes:
    """Descarga archivo desde S3/MinIO."""
    client = get_s3_client()
    response = client.get_object(Bucket=settings.s3_bucket, Key=object_key)
    return response["Body"].read()


def delete_file(object_key: str) -> bool:
    """Elimina archivo de S3/MinIO."""
    client = get_s3_client()
    try:
        client.delete_object(Bucket=settings.s3_bucket, Key=object_key)
        return True
    except ClientError:
        return False
