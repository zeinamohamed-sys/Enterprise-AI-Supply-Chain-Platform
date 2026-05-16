from __future__ import annotations
from pathlib import Path
import random
from typing import Dict, List
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SEED = 42

def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR

def generate_products() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    category_config = {
        "Electronics": {
            "suppliers": ["NovaTech Supply", "Apex Circuits", "Quantum Devices"],
            "price_range": (120.0, 980.0),
            "product_labels": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Gaming Mouse", "4K Webcam", "Portable SSD"],
        },
        "Apparel": {
            "suppliers": ["Urban Thread Co.", "Peak Apparel Group", "LoomLine Partners"],
            "price_range": (18.0, 140.0),
            "product_labels": ["Performance Hoodie", "Training Joggers", "Everyday T-Shirt", "Insulated Jacket", "Running Shorts", "Premium Denim"],
        },
        "Grocery": {
            "suppliers": ["FreshRoute Foods", "Harvest Lane Distribution", "Prime Pantry Supply"],
            "price_range": (4.0, 28.0),
            "product_labels": ["Organic Coffee Beans", "Protein Snack Pack", "Olive Oil Bottle", "Granola Mix", "Green Tea Box", "Sparkling Water Case"],
        },
        "Home": {
            "suppliers": ["HavenNest Wholesale", "Vertex Home Goods", "Northern Living Supply"],
            "price_range": (24.0, 320.0),
            "product_labels": ["Air Purifier", "Desk Lamp", "Storage Organizer", "Bedding Set", "Coffee Table Tray", "Kitchen Blender"],
        },
        "Beauty": {
            "suppliers": ["Radiant Source Labs", "Velvet Care Supply", "PureGlow Distribution"],
            "price_range": (12.0, 95.0),
            "product_labels": ["Hydrating Serum", "Daily Cleanser", "Vitamin C Cream", "Hair Repair Mask", "Sunscreen Lotion", "Body Care Kit"],
        },
        "Sports": {
            "suppliers": ["Momentum Sports Supply", "Endurance Gear House", "Peak Motion Wholesale"],
            "price_range": (22.0, 260.0),
            "product_labels": ["Yoga Mat", "Resistance Bands", "Dumbbell Set", "Foam Roller", "Cycling Bottle", "Training Backpack"],
        },
    }

    records: List[Dict[str, object]] = []
    product_index = 1
    for category, config in category_config.items():
        suppliers = config["suppliers"]
        low, high = config["price_range"]
        labels = config["product_labels"]
        for label in labels:
            supplier = str(rng.choice(suppliers))
            base_price = float(np.round(rng.uniform(low, high), 2))
            records.append({
                "Product_ID": f"P{product_index:03d}",
                "Product_Name": label,
                "Category": category,
                "Base_Price": base_price,
                "Supplier": supplier,
            })
            product_index += 1
    return pd.DataFrame(records)

def _category_seasonality(category: str, current_date: pd.Timestamp) -> float:
    month = current_date.month
    if category == "Electronics": return 1.35 if month in [11, 12] else (0.92 if month in [1, 2] else 1.05)
    if category == "Apparel": return 1.18 if month in [3, 4, 9, 10] else (0.94 if month in [1, 7] else 1.02)
    if category == "Grocery": return 1.12 if month in [11, 12] else 1.00
    if category == "Home": return 1.15 if month in [5, 6, 7] else (0.90 if month in [1, 2] else 1.01)
    if category == "Beauty": return 1.10 if month in [5, 11, 12] else 1.00
    if category == "Sports": return 1.25 if month in [4, 5, 6, 7] else (0.88 if month in [12, 1] else 1.03)
    return 1.0

def _discount_rate(category: str, current_date: pd.Timestamp, rng: np.random.Generator) -> float:
    base_discount = {"Electronics": 0.10, "Apparel": 0.12, "Grocery": 0.05, "Home": 0.08, "Beauty": 0.09, "Sports": 0.11}[category]
    seasonal_boost = 0.04 if current_date.month in [11, 12] else 0.0
    if current_date.weekday() in [5, 6]: seasonal_boost += 0.02
    discount = base_discount + seasonal_boost + float(rng.normal(0.0, 0.015))
    return float(np.clip(round(discount, 4), 0.0, 0.30))

def generate_sales(products_df: pd.DataFrame, customer_count: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=364)
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    products = products_df.copy()
    products["Category_Popularity"] = products["Category"].map({"Electronics": 1.05, "Apparel": 1.00, "Grocery": 1.20, "Home": 0.95, "Beauty": 0.90, "Sports": 1.00})
    baseline_weights = (1.0 / np.sqrt(products["Base_Price"].values)) * products["Category_Popularity"].values
    baseline_weights /= baseline_weights.sum()

    records: List[Dict[str, object]] = []
    order_number = 1

    for current_date in all_dates:
        weekend_boost = 8 if current_date.weekday() in [5, 6] else 0
        holiday_boost = 10 if current_date.month in [11, 12] else 0
        daily_order_count = max(int(rng.poisson(40 + weekend_boost + holiday_boost)), 18)

        seasonal_weights = [_category_seasonality(str(row["Category"]), current_date) for _, row in products.iterrows()]
        effective_weights = baseline_weights * np.array(seasonal_weights)
        effective_weights /= effective_weights.sum()

        for _ in range(daily_order_count):
            product_idx = int(rng.choice(products.index.to_numpy(), p=effective_weights))
            product = products.loc[product_idx]
            quantity = int(rng.choice([1, 2, 3, 4, 5], p=[0.48, 0.26, 0.15, 0.08, 0.03]))
            customer_id = f"C{int(rng.integers(1, customer_count + 1)):04d}"
            discount = _discount_rate(str(product["Category"]), current_date, rng)
            total_price = round(quantity * float(product["Base_Price"]) * (1.0 - discount), 2)

            records.append({
                "Order_ID": f"ORD{order_number:07d}",
                "Date": current_date.strftime("%Y-%m-%d"),
                "Product_ID": str(product["Product_ID"]),
                "Customer_ID": customer_id,
                "Quantity": quantity,
                "Discount": discount,
                "Total_Price": total_price,
            })
            order_number += 1

    return pd.DataFrame(records)

def generate_inventory(products_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    sales = sales_df.copy()
    sales["Date"] = pd.to_datetime(sales["Date"])
    all_dates = pd.date_range(sales["Date"].min(), sales["Date"].max(), freq="D")

    daily_demand = sales.groupby(["Date", "Product_ID"])["Quantity"].sum().unstack(fill_value=0).reindex(index=all_dates, columns=products_df["Product_ID"], fill_value=0)
    records: List[Dict[str, object]] = []

    for _, product in products_df.iterrows():
        product_id = str(product["Product_ID"])
        avg_daily_demand = float(max(daily_demand[product_id].mean(), 1.0))
        lead_time_days = int(rng.integers(5, 12))
        reorder_point = max(int(np.ceil(avg_daily_demand * lead_time_days + (avg_daily_demand * 3.0))), 20)
        stock_level = int(reorder_point * 3 + rng.integers(25, 80))

        for current_date in all_dates:
            sold_units = int(daily_demand.at[current_date, product_id])
            stock_level = max(stock_level - sold_units, 0)
            if stock_level < reorder_point:
                stock_level += int(reorder_point * 2 + rng.integers(30, 120))
            records.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Product_ID": product_id,
                "Stock_Level": stock_level,
                "Reorder_Point": reorder_point,
            })
    return pd.DataFrame(records)

def _generate_review_text(category: str, rating: int, rng: np.random.Generator) -> str:
    pos = [f"This {category.lower()} item delivered excellent quality.", f"I was impressed by the performance and reliability of this {category.lower()}."]
    neu = [f"This {category.lower()} product was acceptable, but it did not stand out.", f"The item works as expected, although it felt average."]
    neg = [f"This {category.lower()} item looked promising, but the performance was disappointing.", f"I expected better durability from this {category.lower()}."]
    return str(rng.choice(pos if rating >= 4 else (neu if rating == 3 else neg)))

def generate_customer_reviews(products_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    purchase_pairs = sales_df.groupby("Product_ID")["Customer_ID"].apply(lambda cs: list(set(cs))).to_dict()
    records: List[Dict[str, object]] = []

    for _, product in products_df.iterrows():
        product_id = str(product["Product_ID"])
        category = str(product["Category"])
        customer_pool = purchase_pairs.get(product_id, [])
        if not customer_pool: continue
        review_count = min(max(len(customer_pool) // 4, 8), 24)
        sampled_customers = rng.choice(customer_pool, size=review_count, replace=True)
        for cust_id in sampled_customers:
            rating = int(rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.12, 0.20, 0.35, 0.25]))
            records.append({
                "Product_ID": product_id,
                "Customer_ID": cust_id,
                "Review_Text": _generate_review_text(category, rating, rng),
                "Rating": rating,
            })
    return pd.DataFrame(records)

def generate_all_data() -> Dict[str, pd.DataFrame]:
    ensure_data_dir()
    p = generate_products()
    s = generate_sales(p)
    i = generate_inventory(p, s)
    r = generate_customer_reviews(p, s)
    p.to_csv(DATA_DIR / "dim_products.csv", index=False)
    s.to_csv(DATA_DIR / "fact_sales.csv", index=False)
    i.to_csv(DATA_DIR / "fact_inventory.csv", index=False)
    r.to_csv(DATA_DIR / "customer_reviews.csv", index=False)
    return {"dim_products": p, "fact_sales": s, "fact_inventory": i, "customer_reviews": r}

if __name__ == "__main__":
    generate_all_data()
  
