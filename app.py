import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

st.title("Customer Segmentation Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write(df.head())

    # Fix missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Select numeric columns
    features = df.select_dtypes(include=['int64','float64'])

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    if st.button("Run Agglomerative Clustering"):

        model = AgglomerativeClustering(n_clusters=5)

        clusters = model.fit_predict(scaled_data)

        df["Cluster"] = clusters

        st.subheader("Clustered Data")
        st.write(df.head())

        st.subheader("Cluster Distribution")
        st.bar_chart(df["Cluster"].value_counts())