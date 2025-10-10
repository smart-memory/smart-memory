import boto3
from botocore.exceptions import ClientError
from typing import Any, Dict, List

from smartmemory.models.memory_item import MemoryItem


class S3Handler:
    """
    Handler for AWS S3 resources (s3://bucket/key URIs).
    All methods accept and return canonical MemoryItem objects or dicts with at least 'content' and 'item_id'.
    Supports get, add (upload), update (overwrite), search (list), and delete.
    Requires AWS credentials to be configured in the environment or via boto3 config.
    """

    def __init__(self):
        self.s3 = boto3.client('s3')

    def _parse_s3_uri(self, uri: str):
        assert uri.startswith('s3://')
        _, _, bucket_and_key = uri.partition('s3://')
        bucket, _, key = bucket_and_key.partition('/')
        return bucket, key

    def get(self, item: 'MemoryItem', **kwargs) -> Dict[str, Any]:
        """
        Retrieve file content from S3 as a MemoryItem/dict. Accepts S3 URI, MemoryItem, or dict.
        Returns a dict with 'content', 'item_id', and 'metadata'.
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('S3Handler only accepts MemoryItem objects')
        uri = item.metadata.get('uri') or item.item_id
        item_id = item.item_id
        bucket, key = self._parse_s3_uri(uri)
        try:
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            content = obj['Body'].read().decode(kwargs.get('encoding', 'utf-8'))
            return {'content': content, 'item_id': item_id, 'metadata': {'uri': uri, 'bucket': bucket, 'key': key}}
        except ClientError as e:
            raise RuntimeError(f"S3 get failed: {e}")

    def add(self, item: 'MemoryItem', **kwargs) -> str:
        """
        Add (upload) a file to S3. Accepts MemoryItem or dict with 'content' and 'item_id' (or 'uri').
        Returns the S3 URI (item_id).
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('S3Handler only accepts MemoryItem objects')
        uri = item.metadata.get('uri') or item.item_id
        content = item.content
        bucket, key = self._parse_s3_uri(uri)
        try:
            self.s3.put_object(Bucket=bucket, Key=key, Body=content.encode(kwargs.get('encoding', 'utf-8')))
            return uri
        except ClientError as e:
            raise RuntimeError(f"S3 add failed: {e}")

    def update(self, item: 'MemoryItem', **kwargs) -> str:
        """
        Update (overwrite) a file in S3. Accepts MemoryItem or dict with 'content' and 'item_id' (or 'uri').
        Returns the S3 URI (item_id).
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('S3Handler only accepts MemoryItem objects')
        return self.add(item, **kwargs)

    def search(self, query: str, item: 'MemoryItem', **kwargs) -> List[str]:
        """
        List all keys in the bucket that contain the query string. Accepts S3 URI, MemoryItem, or dict.
        Returns a list of S3 keys.
        """
        if isinstance(item, str):
            uri = item
        elif MemoryItem is not None and isinstance(item, MemoryItem):
            uri = item.metadata.get('uri') or item.item_id
        else:
            uri = item.get('metadata') or {}.get('uri') or item.get('item_id') or item.get('uri')
        bucket, prefix = self._parse_s3_uri(uri)
        try:
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            raise RuntimeError(f"S3 search failed: {e}")

    def delete(self, item: 'MemoryItem', **kwargs) -> bool:
        """
        Delete a file from S3. Accepts S3 URI, MemoryItem, or dict.
        Returns True on success.
        """
        if isinstance(item, str):
            uri = item
        elif MemoryItem is not None and isinstance(item, MemoryItem):
            uri = item.metadata.get('uri') or item.item_id
        else:
            uri = item.get('metadata') or {}.get('uri') or item.get('item_id') or item.get('uri')
        bucket, key = self._parse_s3_uri(uri)
        try:
            self.s3.delete_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            raise RuntimeError(f"S3 delete failed: {e}")
