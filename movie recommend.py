import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Convert text data into vectors
cv = CountVectorizer()
vectors = cv.fit_transform(df['genre'])

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    if movie not in df['title'].values:
        print("Movie not found!")
        return

    index = df[df['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\nRecommended movies for '{movie}':\n")
    for i in movies_list:
        print(df.iloc[i[0]].title)

# Test
movie = input("Enter movie name: ")
recommend(movie)