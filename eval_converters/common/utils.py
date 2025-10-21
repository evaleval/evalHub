from datetime import datetime

def convert_timestamp_to_unix_format(timestamp: str):
    dt = datetime.fromisoformat(timestamp)
    return str(dt.timestamp())