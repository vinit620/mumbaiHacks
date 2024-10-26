import json
import random


def read_json_file(file_path):
    with open(file_path, "r") as file:
        records = json.load(file)
    return records


def generate_random_pan():
    options = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    length = 10
    pan = "".join(random.choices(options, k=length))

    return pan


def update_pan(records):
    for record in records:
        record["pan_no"] = generate_random_pan()
    return records


def write_json_file(file_path, records):
    with open(file_path, "w") as file:
        json.dump(records, file, indent=4)


if __name__ == "__main__":
    file_path = "MOCK_DATA.json"
    records = read_json_file(file_path)
    updated_records = update_pan(records)
    write_json_file("MOCK_DATA_1.json", updated_records)
