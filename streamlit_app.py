import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("K-Means Clustering App")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:", df.head())

    # Numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns for clustering.")
    else:
        # Pick features
        x_col = st.selectbox("Choose X-axis column", numeric_cols)
        y_col = st.selectbox("Choose Y-axis column", numeric_cols, index=1)

        # Number of clusters
        k = st.slider("Number of clusters (k)", 1, 10, 3)

        # Prepare data
        X = df[[x_col, y_col]]

        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Show clustered data
        st.write("### Clustered Data Preview:", df.head())

        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(df[x_col], df[y_col], c=df['Cluster'])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("K-Means Clustering")

        st.pyplot(fig)

else:
    st.info("Upload a CSV file to begin.")
