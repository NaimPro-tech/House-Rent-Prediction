import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "area_locality": "Bangalore",
    "BHK": 2,
    "Size": 1200,
    "Bathroom": 2
}

res = requests.post(url, json=data)

print(res.json())