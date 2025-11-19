import redis
import json
import os
from typing import Optional, Dict

# Redis client for task storage
def get_redis_client():
    """Get Redis client with environment configuration"""
    try:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_db = int(os.getenv('REDIS_DB', 0))
        redis_password = os.getenv('REDIS_PASSWORD', None)

        # Remove empty password
        if redis_password == '':
            redis_password = None

        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,  # Automatically decode responses to strings
            socket_connect_timeout=5,
            socket_timeout=5
        )

        # Test connection
        client.ping()
        print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        print("ℹ️  Falling back to in-memory storage")
        # Fallback to in-memory storage if Redis is not available
        return None

# Initialize Redis client
redis_client = get_redis_client()

# Fallback in-memory storage (only used if Redis is not available)
task_status: Dict[str, Dict] = {}

# Task storage functions using Redis
def set_task_status(task_id: str, status: dict):
    """Set task status in Redis"""
    if redis_client:
        try:
            redis_client.set(f"task:{task_id}", json.dumps(status))
            # Set expiration to 24 hours to prevent accumulation
            redis_client.expire(f"task:{task_id}", 86400)
        except Exception as e:
            print(f"Failed to set task status in Redis: {e}")
    else:
        # Fallback to in-memory storage
        task_status[task_id] = status

def get_task_status_from_storage(task_id: str) -> Optional[dict]:
    """Get task status from Redis"""
    if redis_client:
        try:
            status_json = redis_client.get(f"task:{task_id}")
            return json.loads(status_json) if status_json else None
        except Exception as e:
            print(f"Failed to get task status from Redis: {e}")
            return None
    else:
        # Fallback to in-memory storage
        return task_status.get(task_id)

def get_all_task_statuses() -> Dict[str, dict]:
    """Get all task statuses from Redis"""
    if redis_client:
        try:
            # Get all task keys
            task_keys = redis_client.keys("task:*")
            tasks = {}
            for key in task_keys:
                task_id = key.replace("task:", "")
                status_json = redis_client.get(key)
                if status_json:
                    tasks[task_id] = json.loads(status_json)
            return tasks
        except Exception as e:
            print(f"Failed to get all task statuses from Redis: {e}")
            return {}
    else:
        # Fallback to in-memory storage
        return task_status