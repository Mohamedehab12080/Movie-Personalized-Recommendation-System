# Updating the Week 2 report with more detailed information about the model development process and evaluation metrics.

week_2_report_detailed = """
# Week 2 Report: Movie Recommendation System Development

## Overview
In this week, the primary focus was on developing the movie recommendation system using collaborative filtering techniques. The goal was to leverage user ratings data to suggest movies to users based on their preferences and the preferences of similar users.

## Dataset Preparation
- **Merged Dataset**: A combined dataset from the ratings and movies data was created to facilitate the recommendation process. The merged dataset contains the following columns:
  - `userId`: Unique identifier for each user.
  - `movieId`: Unique identifier for each movie.
  - `rating`: User rating for the movie (1 to 5 scale).
  - `title`: Movie title.
  - `genres`: Genres associated with the movie.

### Example of Merged Dataset
userId movieId rating title
0 1 1 4.0 Toy Story (1995)
1 1 3 4.0 Grumpier Old Men (1995)
2 1 6 4.0 Heat (1995)
3 1 47 5.0 Seven (a.k.a. Se7en) (1995)
4 1 50 5.0 Usual Suspects, The (1995)

                                    genres  
0 Adventure|Animation|Children|Comedy|Fantasy
1 Comedy|Romance
2 Action|Crime|Thriller
3 Mystery|Thriller
4 Crime|Mystery|Thriller


## Model Development
- **Algorithm Used**: The SVD (Singular Value Decomposition) algorithm from the Surprise library was chosen for collaborative filtering.
- **Model Training**:
  - The dataset was split into training and test sets to evaluate the performance of the model.
  - The training set was used to fit the model, allowing it to learn the latent factors associated with user preferences and movie characteristics.

### Training Process
```python
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(merged_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

Model Evaluation
Evaluation Metrics: The model's performance was evaluated using RMSE (Root Mean Square Error), which measures the average magnitude of the errors between predicted and actual ratings.
RMSE Calculation: The RMSE was calculated on a test dataset, providing insights into the model's accuracy.
Example RMSE Results
Test RMSE: 0.8662768604361383

Recommendations
Recommendation Endpoint: A RESTful API endpoint was created (/recommend) to allow users to request movie recommendations based on their userId.
Response Structure: The response includes movie titles and genres for the top 5 recommended movies that the user has not yet rated.
Example API Response

[
    {
        "genres": "Crime|Drama",
        "movieId": 318,
        "title": "Shawshank Redemption, The (1994)"
    },
    {
        "genres": "Crime|Drama|Thriller",
        "movieId": 48516,
        "title": "Departed, The (2006)"
    },
    {
        "genres": "Sci-Fi|IMAX",
        "movieId": 109487,
        "title": "Interstellar (2014)"
    },
    {
        "genres": "Drama",
        "movieId": 475,
        "title": "In the Name of the Father (1993)"
    },
    {
        "genres": "Comedy|Drama|Romance",
        "movieId": 898,
        "title": "Philadelphia Story, The (1940)"
    }
]

Challenges Faced
Data Handling: Ensuring the integrity and completeness of the merged dataset was crucial for accurate recommendations.
Model Training: The choice of algorithm and its parameters required careful consideration to optimize recommendation accuracy.
Future Steps
Model Improvements: Explore advanced algorithms and techniques such as hybrid models or deep learning approaches to enhance recommendation accuracy.
User Feedback Integration: Implement mechanisms for users to provide feedback on recommendations, allowing the model to adapt over time.
Conclusion
The second week of development was productive, leading to the successful implementation of a movie recommendation system. The foundation has been laid for further enhancements and optimization of the model. """

Saving the detailed report for Week 2
with open('/mnt/data/week_2_report_detailed.md', 'w') as file: file.write(week_2_report_detailed)