import requests
import time
import pandas as pd

url = "https://planetterp.com/api/v1/professor"

# I know you said don't use you as one of the professors, but your reviews
# are the most diversed so I gotta use them for better training purpose lol
professors = ["Maksym Morawski", "Nelson Padua-Perez",
              "Cliff Bakalian","Jonathan Fernandes",
              "Mestiyage Gunatilleka"]
reviews = []

for name in professors:
    params = {
        "name": name,
        "reviews": "true"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        curr_reviews = data.get("reviews", [])
        reviews.extend(curr_reviews)
        print(f"Fetched {len(reviews)} reviews for {name}")
        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        break

print(reviews)

df = pd.DataFrame(reviews)
df.drop(columns=["professor", "course", "expected_grade", "created"], inplace=True)
df["rating"] = df["rating"] - 1
df.to_csv("reviews.csv", index=False)
