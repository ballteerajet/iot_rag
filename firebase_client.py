import requests
import json

BASE_URL = "https://preproject-87b69-default-rtdb.asia-southeast1.firebasedatabase.app/"

def get_sensor_data():
    url = BASE_URL + "sensor_logs.json"
    response = requests.get(url)

    data = response.json()
    print(json.dumps(data, indent=2))

    return data