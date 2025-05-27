#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 21:55:31 2025

@author: chenhungwei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 2025
Author: chenhungwei
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =============================
# Part 1: Load Data and Perform PCA
# =============================
df_total = pd.read_excel("total data 1.xlsx")

scaler = StandardScaler()
X = df_total[["satisfaction mean", "worthwhile mean", "happiness mean"]]
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=1)
wellbeing_pc1 = pca.fit_transform(X_scaled)
df_total["Wellbeing_PC1"] = wellbeing_pc1.round(2)

# =============================
# Part 2: Dashboard Header and Metrics
# =============================
st.title("Crime vs Wellbeing Dashboard (2022 Q1 – 2024 Q3)")

selected_quarter = st.selectbox("Select a Quarter", df_total["Quarter"].unique(), key="quarter_selector")
row = df_total[df_total["Quarter"] == selected_quarter].iloc[0]

st.metric("Crime Total", int(row["crime_total"]))
st.metric("Happiness Mean", round(row["happiness mean"], 2))
st.metric("Satisfaction Mean", round(row["satisfaction mean"], 2))
st.metric("Worthwhile Mean", round(row["worthwhile mean"], 2))
st.metric("Wellbeing_PC1", round(row["Wellbeing_PC1"], 2))

# =============================
# Part 3: Regression Plot
# =============================
st.subheader("Regression: Crime Total vs Happiness")
fig, ax = plt.subplots()
sns.regplot(data=df_total, x="crime_total", y="happiness mean", ax=ax,
            scatter_kws={"color": "blue"}, line_kws={"color": "red"})
ax.set_title("Crime vs Happiness (with Regression Line)")
ax.set_xlabel("Crime Total (per Quarter)")
ax.set_ylabel("Happiness Mean")
st.pyplot(fig)

# =============================
# Part 4: Quarter Comparison
# =============================
st.subheader("Comparison with Another Quarter")
selected_quarter_2 = st.selectbox("Select Another Quarter", df_total["Quarter"].unique(), key="quarter2_compare")
row_2 = df_total[df_total["Quarter"] == selected_quarter_2].iloc[0]

st.metric("Crime Total (Q2)", int(row_2["crime_total"]))
st.metric("Happiness Mean (Q2)", round(row_2["happiness mean"], 2))
st.metric("Wellbeing_PC1 (Q2)", round(row_2["Wellbeing_PC1"], 2))

# =============================
# Part 5: Crime Type and Quarterly Trends
# =============================
st.subheader("Crime Type Distribution & Quarterly Trend")

df_crime = pd.read_csv("cleaned_crime_data.csv")
df_crime.columns = df_crime.columns.str.strip().str.lower()
df_crime["month"] = pd.to_datetime(df_crime["month"])
df_crime["quarter"] = df_crime["month"].dt.to_period("Q")

# Crime type distribution map
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.countplot(data=df_crime, y="crime type", order=df_crime["crime type"].value_counts().index, ax=ax1)
ax1.set_title("Distribution of Types of Offences")
st.pyplot(fig1)

# Quarterly total crime trend chart
crime_per_q = df_crime.groupby("quarter").size().reset_index(name="crime_count")
crime_per_q["quarter"] = crime_per_q["quarter"].astype(str)

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=crime_per_q, x="quarter", y="crime_count", marker="o", ax=ax2)
ax2.set_title("The Quarterly Trend of the Total Number of Offences")
ax2.set_xlabel("Quarter")
ax2.set_ylabel("Crime Count")
plt.xticks(rotation=45)
st.pyplot(fig2)

# =============================
# Part 6: Crime Map
# =============================
with st.expander("Crime Location Map (click to expand)"):
    st.subheader("Crime Location Map")

    sample_df = df_crime.sample(n=min(500, len(df_crime)), random_state=42)

    m = folium.Map(location=[sample_df["latitude"].mean(), sample_df["longitude"].mean()], zoom_start=11)

    for i, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="red",
            fill=True,
            fill_opacity=0.5,
            popup=row["crime type"]
        ).add_to(m)

    st_folium(m, width=700, height=500)

# =============================
# Part 7: Time Series Analysis
# =============================
st.subheader("Time Series Analysis of Key Indicators")
df_total_ts = df_total.copy()
df_total_ts["date"] = pd.date_range(start="2022-01-01", periods=len(df_total_ts), freq="Q")
df_total_ts.set_index("date", inplace=True)

# Trend lines
st.write("### Trend Lines of Wellbeing and Crime Indicators")
fig, ax = plt.subplots(figsize=(10, 6))
for col in ["satisfaction mean", "worthwhile mean", "happiness mean", "Wellbeing_PC1"]:
    ax.plot(df_total_ts.index, df_total_ts[col], marker='o', label=col)
ax.set_title("Time Series Trends of Wellbeing Indicators")
ax.set_ylabel("Score")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df_total_ts.index, df_total_ts["crime_total"], marker='o', color='red')
ax2.set_title("Crime Total Time Series Trend")
ax2.set_ylabel("Crime Total")
plt.xticks(rotation=45)
st.pyplot(fig2)

# Seasonal decomposition
st.write("### Seasonal Decomposition of Wellbeing_PC1 and Crime Total")
for col in ["Wellbeing_PC1", "crime_total"]:
    result = seasonal_decompose(df_total_ts[col], model="additive", period=4)
    fig3 = result.plot()
    fig3.set_size_inches(10, 6)
    st.pyplot(fig3)

# Rolling statistics
st.write("### Rolling Mean and Standard Deviation (window=3)")
for col in ["Wellbeing_PC1", "crime_total"]:
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(df_total_ts[col], label="Original")
    ax4.plot(df_total_ts[col].rolling(3).mean(), label="Rolling Mean (3)")
    ax4.plot(df_total_ts[col].rolling(3).std(), label="Rolling Std (3)")
    ax4.set_title(f"{col} – Rolling Analysis")
    ax4.legend()
    st.pyplot(fig4)