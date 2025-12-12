import requests
import time
import pandas as pd
# url for the planetterp API
url = "https://planetterp.com/api/v1/professor"

# I know you said don't use you for this project, but your reviews
# are the most diversed so I gotta use them for better training purpose lol
professors = ["Maksym Morawski", "Nelson Padua-Perez",
              "Cliff Bakalian", "Jonathan Fernandes",
              "Mestiyage Gunatilleka"]
reviews = []

# fetch reviews for each professor from planetterp
for name in professors:
    params = {
        "name": name,
        "reviews": "true"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        # extract the data and store it
        data = response.json()

        curr_reviews = data.get("reviews", [])
        reviews.extend(curr_reviews)
        print(f"Fetched {len(reviews)} reviews for {name}")

        # wait 2 seconds between each call, hopefully we don't kill the planetterp API again
        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        break

# store the data to csv file
df = pd.DataFrame(reviews)
df.to_csv("reviews.csv", index=False)
