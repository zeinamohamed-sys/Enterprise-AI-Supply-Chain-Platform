# Enterprise AI Analytics & Supply Chain Intelligence Platform

An end-to-end Machine Learning ecosystem and Data Engineering pipeline designed to optimize multi-category retail supply chains. The platform generates synthesized behavioral data, processes multi-faceted ML engines, exposes microservice APIs, and serves interactive executive operational dashboards.

## 🏗️ System Architecture

```text
[Raw Data Engine] ──> (Automated Data Manager) ──> [dim_products / fact_sales CSVs]
                                                          │
                                                          ▼
[FastAPI Microservice] <── [Saved Models] <── (ML Engine: Forecaster/Recommender/NLP)
          │
          ▼
[Streamlit Enterprise UI Dashboard] ──> (What-If Optimization Sandbox)
