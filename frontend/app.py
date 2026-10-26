"""Streamlit frontend: pick a user, call the API, show the "Buy it again" shelf.

Run locally:  streamlit run frontend/app.py   (the API must be running)
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 10

st.set_page_config(page_title="Buy it again", page_icon="🛒", layout="wide")


def call_api(method: str, path: str, **kwargs):
    """Call the API. Return (data, error message)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException:
        return None, f"Cannot reach the API at {API_URL}. Is it running?"
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:  # the error body is not JSON (e.g. an HTML error page)
            detail = None
        return None, detail or f"API error {response.status_code}"
    return response.json(), None


st.title("🛒 Buy it again")
st.caption("Predicts which products an Instacart customer will reorder in their next order.")

health, error = call_api("GET", "/health")
with st.sidebar:
    st.header("API status")
    if error:
        st.error(error)
        st.stop()
    st.success("API is up")
    st.write(f"**Model:** {health['model_name']}")
    st.write(f"**Threshold:** {health['threshold']:.2f}")
    st.write(f"**Users available:** {health['n_users']:,}")
    st.caption(f"API: {API_URL}")

if "user_id" not in st.session_state:
    users, _ = call_api("GET", "/users/sample", params={"n": 1})
    st.session_state.user_id = users[0] if users else 3

col_input, col_k, col_button = st.columns([2, 2, 1])
if col_button.button("🎲 Random user", width="stretch"):
    users, _ = call_api("GET", "/users/sample", params={"n": 1})
    if users:
        st.session_state.user_id = users[0]
user_id = col_input.number_input("User ID", min_value=1, step=1, key="user_id")
k = col_k.slider("Max products on the shelf", 1, 30, 10)

prediction, error = call_api("POST", "/predict", json={"user_id": int(user_id), "k": k})
if error:
    st.warning(
        f"{error}. Only the {health['n_users']:,} users whose next order is unknown (Kaggle test set) "
        "can get recommendations. Click 🎲 Random user to pick one."
    )
    st.stop()
history, _ = call_api("GET", f"/users/{int(user_id)}/history", params={"k": 10})

left, right = st.columns(2)
with left:
    st.subheader("Recommended: buy it again")
    items = pd.DataFrame(prediction["items"])
    if items.empty:
        st.info("The model recommends nothing: this user will probably not reorder any past product.")
    else:
        st.dataframe(
            items[["product_name", "department", "probability"]],
            # fixed widths, so a long product name does not push the other columns out of view
            column_config={
                "product_name": st.column_config.TextColumn("Product", width="medium"),
                "department": st.column_config.TextColumn("Department", width="small"),
                "probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="%.2f", width="small"
                ),
            },
            hide_index=True,
            width="stretch",
        )
    st.caption(
        f"Showing {len(items)} products with probability >= {prediction['threshold']:.2f} (at most {k}), "
        f"out of {prediction['n_candidates']} past products."
    )

with right:
    st.subheader("Most bought in the past")
    past = pd.DataFrame(history or [])
    if not past.empty:
        st.dataframe(
            past[["product_name", "department", "times_bought", "orders_since_last"]],
            column_config={
                "product_name": st.column_config.TextColumn("Product", width="medium"),
                "department": st.column_config.TextColumn("Department", width="small"),
                "times_bought": st.column_config.NumberColumn("Times bought", width="small"),
                "orders_since_last": st.column_config.NumberColumn(
                    "Orders ago", help="Orders since the last purchase (0 = in the last order)", width="small"
                ),
            },
            hide_index=True,
            width="stretch",
        )
