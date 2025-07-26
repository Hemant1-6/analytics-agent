from cachetools import TTLCache

# In-memory cache for up to 10 datasets, with each entry living for 1 hour (3600s)
DATASET_CACHE = TTLCache(maxsize=10, ttl=3600)