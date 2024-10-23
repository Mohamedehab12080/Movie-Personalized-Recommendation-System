import pandas as pd

# Load ratings and movies data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Display the first few rows of the ratings and movies datasets
print("Ratings Dataset:")
print(ratings.head())

print("\nMovies Dataset:")
print(movies.head())


# Check for missing values in ratings
print("Missing values in ratings dataset:")
print(ratings.isnull().sum())

# Check for missing values in movies
print("Missing values in movies dataset:")
print(movies.isnull().sum())


# Merge the ratings and movies datasets on the 'movieId'
merged_data = pd.merge(ratings, movies, on='movieId')

# Display the first few rows of the merged dataset
print("Merged Dataset:")
print(merged_data.head())

merged_data.to_csv('merged_dataset.csv', index=False)



# Normalize ratings between 0 and 1
merged_data['normalized_rating'] = merged_data['rating'] / 5.0

# Display first few rows
print(merged_data[['userId', 'movieId', 'rating', 'normalized_rating']].head())


from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the ratings dataset into the Surprise library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use the SVD algorithm for collaborative filtering
algo = SVD()

# Train the model on the training data
algo.fit(trainset)

# Make predictions on the test data
predictions = algo.test(testset)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a count matrix for the genres
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(movies['genres'])

# Compute the cosine similarity between movies
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to get movie recommendations based on similarity
def get_recommendations(movie_title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies[movies['title'] == movie_title].index[0]

    # Get the pairwise similarity scores of all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar movies
    movie_indices = [i[0] for i in sim_scores[1:11]]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

# Test the function
print(get_recommendations('Toy Story (1995)'))


from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse  # Correct import for accuracy

# Load dataset
file_path = 'merged_dataset.csv'
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

# Split into training and test set
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize SVD model (or other model of your choice)
model = SVD()

# Train the model
model.fit(trainset)

# Test the model
predictions = model.test(testset)

# Calculate and print RMSE
rmse_value = rmse(predictions)
print(f'Test RMSE: {rmse_value}')


from flask import Flask, request, jsonify
from surprise import Dataset, Reader, SVD
import pandas as pd

# Initialize Flask app
app = Flask(__name__)



# Merge datasets
merged_df = pd.read_csv("merged_dataset.csv")

# Load model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(merged_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD()  # or any other model you want to use
model.fit(trainset)

# Route to get movie recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    # Get userId from query params, with error handling
    user_id_str = request.args.get('userId')
    if not user_id_str:
        return jsonify({"error": "userId parameter is required"}), 400

    try:
        user_id = int(user_id_str)
    except ValueError:
        return jsonify({"error": "Invalid userId. Must be an integer."}), 400

    # Get all movie IDs
    all_movie_ids = merged_df['movieId'].unique()

    # Get the movie IDs the user has already rated
    rated_movie_ids = merged_df[merged_df['userId'] == user_id]['movieId'].values

    # Recommend movies the user hasn't rated yet
    recommendations = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movie_ids:
            pred = model.predict(user_id, movie_id)
            recommendations.append((movie_id, pred.est))

    # Check if there are any recommendations
    if not recommendations:
        return jsonify({"message": "No new recommendations found for user."}), 200

    # Sort recommendations by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:5]  # Top 5 recommendations

    # Get movie titles and genres from the merged dataset
    top_movie_ids = [rec[0] for rec in top_recommendations]
    top_movies = merged_df[merged_df['movieId'].isin(top_movie_ids)].drop_duplicates('movieId')

    return jsonify(top_movies[['movieId', 'title', 'genres']].to_dict(orient='records'))

if __name__ == "__main__":
    # Ensure the Flask app runs in development mode only if this script is executed directly
    app.run(debug=True, use_reloader=False)
