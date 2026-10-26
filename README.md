# Instacart reorder prediction ("Buy it again")

[![CI](https://github.com/NamAnh-dev/Instacart-reorder-prediction-project/actions/workflows/ci.yml/badge.svg)](https://github.com/NamAnh-dev/Instacart-reorder-prediction-project/actions/workflows/ci.yml)

Grocery apps often show a "Buy it again" shelf. In this project I predict which past products an Instacart
customer will buy again in their next order, and serve the predictions with a FastAPI service and a small
Streamlit app.

Stack: pandas, scikit-learn, LightGBM, FastAPI, Streamlit, Docker, pytest, GitHub Actions.

![Streamlit app](reports/figures/app_streamlit.png)

## Problem

Each row is a (user, product) pair for a product the user has bought before. The label is 1 if the user
buys it again in their next order. A model gives every pair a probability, and the shelf shows the products
above a threshold.

The main metric is the mean F1 per order, similar to the Kaggle competition metric but without its special
"None" item: if a user buys nothing again and the shelf is empty, F1 = 1. I also report precision@10 and
recall@10 for a shelf of 10 products.

## Data

[Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) on Kaggle:
3.4M orders from 206k users. Every user has a purchase history and one last order.

- 131,209 users have a known last order. I split them by user into train / validation / test (70 / 15 / 15).
- 75,000 users have a hidden last order (Kaggle test set). They are the users the API makes recommendations for.

## Approach

1. `src/preprocess.py`: quality checks, labels, user split.
2. `src/features.py`: 16 features about the user, the product and the user × product pair
   (how often and how recently it was bought). The same code builds the features for training and for the API.
3. `src/baselines.py`: two simple rules, "repeat the last order" and "products bought in at least X% of orders".
4. `src/models.py`, `src/train.py`: logistic regression, random forest and LightGBM. The threshold of each model
   and the final model are chosen on the validation users; the test users are used once at the end.
5. `notebooks/02_model_results.ipynb`: feature importance and error analysis.

Some choices: I split by user so that the model is always tested on new users. I do not use anything about
the next order (day, hour, days since the last order) because the shelf is built before the customer orders.
Only about 10% of the pairs are positive, so the best threshold is far below 0.5.

## Results

On the 19,682 test users (95% interval from a bootstrap over users):

| System | Setting (chosen on valid) | Test F1 [95% CI] | Precision@10 | Recall@10 | ROC AUC | Avg items |
|---|---|---|---|---|---|---|
| Baseline: repeat last order | products of the last order | 0.312 [0.309, 0.316] | 0.230 | 0.455 | - | 10.3 |
| Baseline: purchase rate | order rate >= 0.26 | 0.327 [0.324, 0.331] | 0.272 | 0.518 | 0.772 | 10.7 |
| Logistic regression | probability >= 0.18 | 0.364 [0.360, 0.367] | 0.300 | 0.575 | 0.825 | 9.5 |
| Random forest | probability >= 0.20 | 0.376 [0.372, 0.380] | 0.306 | 0.584 | 0.832 | 8.8 |
| LightGBM (final) | probability >= 0.21 | 0.376 [0.373, 0.380] | 0.307 | 0.586 | 0.833 | 8.4 |

What I take from it:

- LightGBM beats the best rule by 0.049 F1 (+15%) and shows fewer products (8.4 vs 10.7 per user).
  Random forest is just as good (the intervals overlap); I kept LightGBM because its validation F1 was higher.
- On a 10-product shelf, about 3 of the 10 products are bought again, vs 2.7 with the rule, and on average
  the shelf contains 59% of the products the customer reorders, vs 52%.
- How often and how recently the user bought the product matter most
  (`up_order_rate`, `up_rate_since_first`, `up_orders_since_last`).
- Weak spots: the 7% of users who buy nothing again (F1 about 0.19) and products bought rarely
  (pantry, household).
- These are offline numbers. The effect on real orders would need an A/B test.

## How to run

Python 3.12:

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Data: download the competition data from Kaggle (you need to accept the competition rules) and put the 6 CSV
files in `data/raw/`: `aisles.csv`, `departments.csv`, `order_products__prior.csv`,
`order_products__train.csv`, `orders.csv`, `products.csv`. With the Kaggle CLI:
`kaggle competitions download -c instacart-market-basket-analysis -p data/raw`, then unzip.

Train (about 8 minutes and 6 GB of RAM on my laptop, i5-1035G1 with 16 GB):

```bash
python -m src.pipeline
```

Serve the API (http://localhost:8000/docs) and the app (http://localhost:8501):

```bash
docker compose up --build
```

Without Docker, run `uvicorn api.main:app --port 8000` and `streamlit run frontend/app.py` in two terminals.

Tests and lint (also run by GitHub Actions on every push):

```bash
pytest -q
ruff check .
```

## Project structure

```
src/         preprocess, features, baselines, models, evaluate, train, pipeline, recommender (used by the API)
api/         FastAPI service and its Dockerfile
frontend/    Streamlit app and its Dockerfile
notebooks/   01_eda, 02_model_results
reports/     metrics.json, model_comparison.md, feature_importance.csv, figures/
tests/       tests on a tiny hand-made dataset, API tests
```

## Limitations

- Only products the user bought before can be recommended.
- One threshold for everyone; choosing the shelf size per user would help the users who buy nothing again.
- The data has no real dates, so the split is by user and not by time.
- Serving features are computed once in advance; in production they would be updated after every order.

## How I built this

I scoped the problem, chose the metrics, ran the experiments and analysed the results. I used an AI assistant
(Claude) to help write and review the code, the tests and this README. Every number above comes from
`reports/metrics.json` or the notebook outputs, and re-running the pipeline from the raw CSV files gives the
same numbers.
