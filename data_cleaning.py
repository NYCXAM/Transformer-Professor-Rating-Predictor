import pandas as pd
import re

# 1. Load Data
df = pd.read_csv("reviews.csv")

original_count = len(df)
print(f"Original review count: {original_count}")

# 2. Remove Emojis
# This regex matches most emoji ranges in Unicode
def has_emoji(text):
    if not isinstance(text, str): return False
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))

df = df[~df['review'].apply(has_emoji)]
print(f"Count after removing emojis: {len(df)}")

# 3. Filter Shortest and Longest 5%
# Calculate word count
df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

# Calculate the cutoffs (2nd and 98th percentile)
lower_bound = df['word_count'].quantile(0.02)
upper_bound = df['word_count'].quantile(0.98)

print(f"Keeping reviews with length between {lower_bound} and {upper_bound} words.")

# Filter
df = df[(df['word_count'] >= lower_bound) & (df['word_count'] <= upper_bound)]

# 4. Save
# We drop the helper 'word_count' column before saving
df = df.drop(columns=['word_count'])
df.to_csv("cleaned_reviews.csv", index=False)

print(f"Final clean count: {len(df)}")
print(f"Removed {original_count - len(df)} bad reviews.")
print("Saved to 'cleaned_reviews.csv'")