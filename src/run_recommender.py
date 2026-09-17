from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/"src"))
from recommender import load_data, build_content_model, popularity_scores, user_profile_recommend

restaurants, interactions=load_data(ROOT/"data")
_,_,similarity=build_content_model(restaurants)
print(f"Loaded {len(restaurants)} restaurants and {len(interactions)} interactions.")
user_id=int(interactions.user_id.iloc[0])
print(f"\nRecommendations for user {user_id}:")
print(user_profile_recommend(restaurants,interactions,similarity,user_id,10)[["restaurant_name","cuisine","area","rating"]].to_string(index=False))
