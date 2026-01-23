from prometheus_client import Counter

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
)
