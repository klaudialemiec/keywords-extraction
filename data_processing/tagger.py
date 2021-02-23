import time
from enum import Enum
from pathlib import Path
from config import username
import requests

BASE_URL = "http://ws.clarin-pl.eu/nlprest2/base"
BASE_DIR = Path.cwd()


class DataSource(Enum):
    LOCAL_UNCOMPRESSED_FILE = 0
    LOCAL_COMPRESSED_FILE = 1
    REMOTE_COMPRESSED_FILE = 2
    TEXT = 3


def download(file_id: str, file_path: str) -> None:
    response = get_response(file_id)
    with open(file_path, "w+", encoding="utf-8") as f:
        f.write(response.text)


def get_response(file_id: str) -> str:
    url = f"{BASE_URL}/download{file_id}"
    response = requests.get(url=url)
    return response.text


def upload(file_path: str) -> str:
    url = f"{BASE_URL}/upload/"
    headers = {"content-type": "application/json"}
    with open(file_path, "rb", 'utf-8') as f:
        response = requests.post(url=url, data=f.read(), headers=headers)
    return response.text


def start_task(data):
    url = "{BASE_URL}/startTask/"
    response = requests.post(url=url, json=data)
    task_id = response.text
    time.sleep(0.5)
    return task_id


def get_status(task_id: str) -> dict:
    url = f"{BASE_URL}/getStatus/{task_id}"
    response = requests.get(url=url)
    return response.json()


def process_task(task_id: str) -> str:
    status = get_status(task_id)
    while status["status"] in ("QUEUE", "PROCESSING"):
        time.sleep(0.5)
        status = get_status(task_id)

    if status["status"] == "ERROR":
        raise requests.HTTPError(status["value"])

    return status["value"]


def lemmatize(text: str, user: str = username) -> str:
    data = {"user": user,
            "lpmn": "any2txt|wcrft2({\"morfeusz2\":false})", "text": text}
    task_id = start_task(data)
    items = process_task(task_id=task_id)
    for item in items:
        response = get_response(file_id=item["fileID"])
    return response
