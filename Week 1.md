
# Preprocessing Documentation for Personalized Recommendation System

---

## 1. Introduction
This report documents the data preprocessing steps performed on the MovieLens dataset to prepare it for building a recommendation system. The preprocessing phase is critical to ensure that the data is clean, structured, and ready for machine learning models. We address missing data, merge datasets, and normalize user ratings to achieve this goal.

---

## 2. Data Loading
The MovieLens dataset consists of two primary files:
- `ratings.csv`: Contains user ratings for movies.
- `movies.csv`: Contains metadata about the movies.

The data was loaded using the Pandas library as follows:

```python
import pandas as pd

# Load the ratings and movies data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
```

---

## 3. Handling Missing Data

Missing data can cause issues in machine learning models, so it was essential to check if any missing values existed in both datasets. The following code was used to check for missing data:

```python
# Check for missing values in the ratings and movies datasets
ratings_missing = ratings.isnull().sum()
movies_missing = movies.isnull().sum()

print(ratings_missing)
print(movies_missing)
```

**Results:**
- **Ratings Dataset:** No missing values were found.
- **Movies Dataset:** No missing values were found.

Since no missing values were present in either dataset, no further imputation or removal was necessary.

---

## 4. Merging Datasets

To facilitate content-based recommendations, we needed to merge the ratings dataset with the movie details (i.e., movie titles and genres). The `movieId` was the key used to merge these datasets.

The merged dataset combines user ratings with movie details, allowing us to use the movie's genres for content-based filtering.

```python
# Merge the ratings and movies datasets on 'movieId'
merged_data = pd.merge(ratings, movies, on='movieId')
```

The resulting dataset contains the following columns:
- `userId`: The unique identifier for each user.
- `movieId`: The unique identifier for each movie.
- `rating`: The rating given by the user (on a scale from 0.5 to 5.0).
- `timestamp`: When the rating was given.
- `title`: The movie title.
- `genres`: The genres associated with the movie.

---

## 5. Data Normalization

Normalization is often used to bring all data into a standard range, which can help improve model performance. Since ratings are on a scale from 0.5 to 5, we normalized the ratings by dividing them by 5. This normalization transforms the ratings to a scale from 0 to 1.

```python
# Normalize the ratings column to a 0-1 scale
merged_data['normalized_rating'] = merged_data['rating'] / 5.0
```

After normalization, the `normalized_rating` column provides a consistent scale for the ratings.

---

## 6. Data Exploration

Basic exploration of the ratings distribution was performed to understand the range and frequency of ratings in the dataset. The distribution of ratings is shown below:

```python
import matplotlib.pyplot as plt

# Plot the distribution of ratings
merged_data['rating'].hist(bins=10)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')
plt.show()
```

The ratings are primarily clustered around 3.0 to 4.5, with fewer ratings at the extremes (0.5 and 5.0).

---

## 7. Train-Test Split

To evaluate the recommendation model’s performance, we split the dataset into training and test sets. This ensures that the model is trained on one part of the data and tested on another, preventing overfitting.

```python
from sklearn.model_selection import train_test_split

# Split data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
```

---

## 8. Conclusion

The preprocessing phase successfully prepared the MovieLens dataset for the next phase of building a personalized recommendation system. By handling missing data, merging datasets, and normalizing ratings, the data is now clean and ready for both collaborative filtering and content-based recommendation models. This ensures that the subsequent machine learning models can be built on reliable and consistent data.

---

## Deliverables:
- **Cleaned Dataset**: The preprocessed dataset is ready for machine learning.
- **Documentation**: This report outlines the key steps taken during preprocessing.

---
