import datetime

import pytz


def get_current_timestamp() -> str:
    tz = pytz.timezone("Europe/Berlin")
    timestamp = datetime.datetime.now(tz=tz).isoformat()
    return timestamp


def convert_to_list(items):
    result = items.tolist()

    if isinstance(result, list):
        return result
    else:
        return [result]
