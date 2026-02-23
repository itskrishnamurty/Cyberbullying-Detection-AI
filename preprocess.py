import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load dataset
df = pd.read_csv("data/raw/cyberbullying_tweets.csv")

# initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove emojis
    text = emoji.replace_emoji(text, replace='')

    # remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # tokenize
    words = text.split()

    # remove stopwords & lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

# apply cleaning
df["clean_text"] = df.iloc[:,0].apply(clean_text)

# save cleaned dataset
df.to_csv("data/processed/cleaned_data.csv", index=False)

print("\nSample cleaned text:\n")
print(df[["clean_text"]].head())