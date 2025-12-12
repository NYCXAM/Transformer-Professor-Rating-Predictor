import pandas as pd
import re

df = pd.read_csv("reviews.csv")

# make sure the label start at 0
df["rating"] = df["rating"] - 1
# drop unnecessary columns
df.drop(columns=["professor", "course", "expected_grade", "created"], inplace=True)
# rename the rating column to label
df.rename(columns={"rating": "label"}, inplace=True)

original_count = len(df)
print(f"Original review count: {original_count}")

# remove emojis
def has_emoji(text):
    if not isinstance(text, str): return False
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))

df = df[~df['review'].apply(has_emoji)]
print(f"Count after removing emojis: {len(df)}")

# remove shortest and longest 2% reviews
df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

# calculate the cutoffs
lower_bound = df['word_count'].quantile(0.02)
upper_bound = df['word_count'].quantile(0.98)

print(f"Keeping reviews with length between {lower_bound} and {upper_bound} words.")

df = df[(df['word_count'] >= lower_bound) & (df['word_count'] <= upper_bound)]

# Save
# drop the helper word_count column before saving
df = df.drop(columns=['word_count'])
df.to_csv("cleaned_reviews.csv", index=False)

print(f"Final clean count: {len(df)}")
print(f"Removed {original_count - len(df)} bad reviews.")
print("Saved to 'cleaned_reviews.csv'")