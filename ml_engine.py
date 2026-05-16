from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(DATA_DIR / "dim_products.csv")
    s = pd.read_csv(DATA_DIR / "fact_sales.csv")
    i = pd.read_csv(DATA_DIR / "fact_inventory.csv")
    r = pd.read_csv(DATA_DIR / "customer_reviews.csv")
    return p, s, i, r

def build_forecaster_features() -> Tuple[pd.DataFrame, pd.Series]:
    _, s, _, _ = load_clean_data()
    s["Date"] = pd.to_datetime(s["Date"])
    daily_sales = s.groupby(["Date", "Product_ID"])["Quantity"].sum().reset_index()
    
    daily_sales = daily_sales.sort_values(by=["Product_ID", "Date"]).reset_index(drop=True)
    daily_sales["Lag_1"] = daily_sales.groupby("Product_ID")["Quantity"].shift(1)
    daily_sales["Lag_2"] = daily_sales.groupby("Product_ID")["Quantity"].shift(2)
    daily_sales["Roll_7_Mean"] = daily_sales.groupby("Product_ID")["Quantity"].transform(lambda x: x.shift(1).rolling(7).mean())
    
    daily_sales = daily_sales.dropna().reset_index(drop=True)
    daily_sales["Month"] = daily_sales["Date"].dt.month
    daily_sales["DayOfWeek"] = daily_sales["Date"].dt.dayofweek
    
    feature_cols = ["Lag_1", "Lag_2", "Roll_7_Mean", "Month", "DayOfWeek"]
    X = daily_sales[feature_cols]
    y = daily_sales["Quantity"]
    return X, y

def train_demand_forecaster() -> Tuple[RandomForestRegressor, Dict[str, float]]:
    X, y = build_forecaster_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds))
    }
    return model, metrics

class RuleBasedCollaborativeRecommender:
    def __init__(self) -> None:
        self.association_rules: Dict[str, list] = {}

    def fit(self, sales_df: pd.DataFrame) -> RuleBasedCollaborativeRecommender:
        orders = sales_df.groupby("Order_ID")["Product_ID"].apply(list).to_list()
        co_occurrences: Dict[str, Dict[str, int]] = {}
        
        for basket in orders:
            unique_items = list(set(basket))
            for i in range(len(unique_items)):
                item_a = unique_items[i]
                if item_a not in co_occurrences:
                    co_occurrences[item_a] = {}
                for j in range(len(unique_items)):
                    if i == j: continue
                    item_b = unique_items[j]
                    co_occurrences[item_a][item_b] = co_occurrences[item_a].get(item_b, 0) + 1
                    
        for item, peers in co_occurrences.items():
            sorted_peers = sorted(peers.items(), key=lambda x: x[1], reverse=True)
            self.association_rules[item] = [p[0] for p in sorted_peers[:3]]
        return self

    def recommend(self, product_id: str, top_n: int = 2) -> list:
        return self.association_rules.get(product_id, [])[:top_n]

class StaticRuleSentimentAnalyzer:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.positive_keywords = {"excellent", "impressed", "quality", "performance", "reliability", "good", "great"}
        self.negative_keywords = {"disappointing", "expected", "better", "durability", "bad", "poor", "felt"}

    def fit(self, reviews_df: pd.DataFrame) -> StaticRuleSentimentAnalyzer:
        self.vectorizer.fit(reviews_df["Review_Text"])
        return self

    def predict_sentiment(self, text: str) -> str:
        tokens = text.lower().split()
        pos_count = sum(1 for t in tokens if t in self.positive_keywords)
        neg_count = sum(1 for t in tokens if t in self.negative_keywords)
        if pos_count > neg_count: return "Positive"
        if neg_count > pos_count: return "Negative"
        return "Neutral"

def execute_pipeline() -> Dict[str, Any]:
    p, s, i, r = load_clean_data()
    forecaster, forecast_metrics = train_demand_forecaster()
    
    recommender = RuleBasedCollaborativeRecommender()
    recommender.fit(s)
    
    analyzer = StaticRuleSentimentAnalyzer()
    analyzer.fit(r)
    
    return {
        "forecaster": forecaster,
        "forecast_metrics": forecast_metrics,
        "recommender": recommender,
        "sentiment_analyzer": analyzer
    }

if __name__ == "__main__":
    results = execute_pipeline()
    print("Pipeline executed. Forecaster Metrics:", results["forecast_metrics"])
  
