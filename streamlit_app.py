import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import altair as alt

st.title("K-Means Clustering App (No Matplotlib)")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Please upload a dataset with at least two numeric columns.")
    else:
        x_col = st.selectbox("Select X-axis column", numeric_cols)
        y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1)

        # Choose number of clusters
        k = st.slider("Number of clusters (k)", 1, 10, 3)

        # Prepare data
        X = df[[x_col, y_col]]

        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        st.write("### Clustered Data Preview", df.head())

        # Altair scatter plot
        chart = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(
                x=x_col,
                y=y_col,
                color='Cluster:N',
                tooltip=[x_col, y_col, 'Cluster']
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

else:
    st.info("Upload a CSV file to start.")
