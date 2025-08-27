import os

bind = f"0.0.0.0:{os.environ.get('PORT', '90')}"  
workers = int(os.environ.get("WEB_CONCURRENCY", "4"))
timeout = 120