import os
port = os.getenv("PORT", "90")
bind = f"0.0.0.0:{port}"
workers = os.getenv("WEB_CONCURRENCY", "4")
timeout = 120
