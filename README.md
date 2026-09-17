# Restaurant Recommendation System

An end-to-end recommendation-system project exploring **content-based filtering, implicit user feedback, personalization, cold-start handling, and offline evaluation**.

## Project Overview

Recommendation systems help users discover relevant items from large collections of choices.

In this project, I explored how restaurant metadata and user interaction history can be combined to generate personalized recommendations and how recommendation quality can be evaluated using offline metrics.

## Approach

```text
Restaurant Metadata + User Interactions
                  ↓
          Data Understanding
                  ↓
         Exploratory Analysis
                  ↓
       TF-IDF Representation
                  ↓
          Cosine Similarity
                  ↓
    ┌─────────────┴─────────────┐
    │                           │
New / Cold-Start User      Known User
    │                           │
Popularity Baseline        Weighted Profile
    │                           │
    └─────────────┬─────────────┘
                  ↓
          Top-K Recommendations
                  ↓
           Offline Evaluation
```

## Techniques

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity
* Implicit Feedback
* Popularity-Based Recommendation
* Content-Based Filtering
* Personalized Recommendations
* Precision@10
* Hit Rate@10

## Dataset

The project uses **synthetic data created for experimentation and learning**.

### `restaurants.csv`

Contains information about **300 restaurants**, including:

* Cuisine
* Area
* Price level
* Rating
* Delivery time
* Ambience
* Dietary type

### `user_interactions.csv`

Contains synthetic interactions from **1,000 users**.

The interactions include:

* `view`
* `click`
* `order`

Different interaction types are assigned different weights to represent different levels of user interest.

| Interaction | Weight |
| ----------- | -----: |
| View        |      1 |
| Click       |      3 |
| Order       |      5 |

These weights are assumptions used for experimentation and are not based on real-world platform data.

## Recommendation Approach

### 1. Popularity-Based Recommendation

A popularity baseline ranks restaurants according to their total weighted interaction volume.

This provides a simple baseline and can also be used when sufficient user history is unavailable.

### 2. Content-Based Filtering

Restaurant attributes are combined into a textual representation and converted into numerical vectors using **TF-IDF**.

Cosine similarity is then used to identify restaurants with similar characteristics.

```text
Restaurant Metadata
        ↓
Text Representation
        ↓
      TF-IDF
        ↓
Restaurant Vectors
        ↓
Cosine Similarity
        ↓
Similar Restaurants
```

### 3. Personalized Recommendations

For users with interaction history, the system combines similarity scores from restaurants they previously interacted with.

Interaction weights are used so that stronger interactions contribute more to the user's preference profile.

Restaurants already interacted with by the user are excluded from the final recommendations.

## Evaluation

The recommendation system is evaluated using an offline leave-one-out style experiment.

### Precision@10

Measures the proportion of the top 10 recommendation positions that correspond to the held-out relevant item.

### Hit Rate@10

Measures the proportion of evaluated users for whom the held-out item appears within the top 10 recommendations.

For this synthetic dataset:

```text
Evaluated Users: 1,000
Precision@10: 0.0261
Hit Rate@10: 0.2611
```

These results are specific to the synthetic dataset and experimental setup.

Offline metrics are useful for evaluating recommendation approaches during development, but they do not directly establish real-world user or business impact.

## Project Structure

```text
restaurant-recommendation-system/
│
├── data/
│   ├── restaurants.csv
│   └── user_interactions.csv
│
├── notebooks/
│   └── Restaurant_Recommendation_System.ipynb
│
├── outputs/
│   ├── evaluation_results.csv
│   ├── example_similar_restaurants.csv
│   ├── restaurant_rating_distribution.png
│   ├── top_popular_restaurants.csv
│   └── weighted_interactions_by_cuisine.png
│
├── src/
│   ├── recommender.py
│   └── run_recommender.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/Restaurant_Recommendation_System.ipynb
```

Run the notebook from top to bottom.

The recommendation pipeline can also be executed using:

```bash
python src/run_recommender.py
```

## Key Concepts Explored

* Recommendation Systems
* Content-Based Filtering
* TF-IDF
* Cosine Similarity
* Implicit Feedback
* User Preference Modeling
* Cold-Start Problem
* Popularity Baselines
* Top-K Recommendations
* Precision@K
* Hit Rate@K
* Offline Evaluation

## Limitations

1. The dataset is synthetic.
2. User interactions are simulated rather than collected from real users.
3. Interaction weights are manually defined.
4. The system primarily uses content-based information rather than a full collaborative filtering approach.
5. Evaluation is performed offline.
6. Real-world contextual factors such as location, time, availability, and changing user preferences are not included.

## Possible Improvements

The system could be extended by exploring:

* Collaborative Filtering
* Hybrid Recommendation Models
* User-Item Interaction Matrices
* Context-Aware Recommendations
* Recommendation Diversity
* Popularity Bias Reduction
* Improved Cold-Start Strategies
* Recall@K and NDCG@K
* Candidate Generation and Ranking Architectures
* Online Experimentation

## Learning Outcome

Through this project, I explored the complete workflow of building a recommendation system — from understanding interaction data and representing item attributes to generating personalized recommendations and evaluating recommendation quality using offline metrics.

The project was developed as a hands-on exploration of recommendation-system concepts and their practical implementation in Python.
