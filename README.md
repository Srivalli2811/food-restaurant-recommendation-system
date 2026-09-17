# Food & Restaurant Recommendation System

An end-to-end recommendation-system portfolio project demonstrating **content-based filtering, implicit user feedback, personalization, cold-start handling and offline evaluation**.

## Business problem
Recommend relevant restaurants to users while providing a fallback for users with no history.

## Approach
```text
Restaurant metadata + user interactions
                ↓
          Data validation
                ↓
        Exploratory analysis
                ↓
     TF-IDF restaurant vectors
                ↓
       Cosine similarity
                ↓
  ┌─────────────┴─────────────┐
  │                           │
New / cold-start user     Known user
  │                           │
Popularity fallback      Weighted profile
  └─────────────┬─────────────┘
                ↓
        Top-K recommendations
                ↓
        Offline evaluation
```

## Techniques
- Python, Pandas, NumPy
- Scikit-learn
- TF-IDF and cosine similarity
- Implicit-feedback weighting
- Popularity baseline
- Personalized content-based recommendations
- Precision@10 and Hit Rate@10
- Recommendation-system business analysis

## Dataset
The project uses **synthetic portfolio data**:
- `restaurants.csv`: 300 restaurants with metadata
- `user_interactions.csv`: synthetic views, clicks and orders from 1,000 users

**This is not Swiggy internal data and does not represent Swiggy's actual operations or performance.**

## Structure
```text
food-restaurant-recommendation-system/
├── data/
│   ├── restaurants.csv
│   └── user_interactions.csv
├── notebooks/
│   └── Restaurant_Recommendation_System.ipynb
├── outputs/
├── src/
│   ├── recommender.py
│   └── run_recommender.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Open `notebooks/Restaurant_Recommendation_System.ipynb` and run cells from top to bottom.

Or:
```bash
python src/run_recommender.py
```

## Evaluation
The notebook includes a simple leave-one-out offline experiment using Precision@10 and Hit Rate@10. Offline metrics do not establish business impact; production systems should also monitor ranking quality, coverage, diversity, latency and online A/B-test metrics.

## Limitations and production roadmap
The data is synthetic and lacks real-time location, demand and restaurant availability. A production system could add collaborative filtering, hybrid ranking, contextual signals, two-stage candidate generation/ranking, diversity constraints, cold-start strategies and online experimentation.

## Interview topics
TF-IDF, cosine similarity, implicit feedback, cold start, Precision@K, recommendation architecture, collaborative filtering, diversity, and A/B testing.
