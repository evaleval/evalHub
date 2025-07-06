
import json


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_json(filename) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
