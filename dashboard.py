import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("clean data/main_data.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

df = load_data()

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("📊 Air Quality Dashboard")

stations = st.sidebar.multiselect(
    "Pilih Station:",
    options=df['station'].unique(),
    default=df['station'].unique()
)

df_filtered = df[df['station'].isin(stations)]

# =====================================
# TITLE
# =====================================
st.title("🌍 Air Quality Analysis Dashboard")
st.markdown("Analisis PM2.5 di 12 Stasiun (2013–2017)")

# =====================================
# 1. OVERVIEW
# =====================================
st.subheader("📌 Data Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Total Data", len(df_filtered))
col2.metric("Jumlah Station", df_filtered['station'].nunique())
col3.metric("Rata-rata PM2.5", round(df_filtered['PM2.5'].mean(),2))

# =====================================
# 2. PM2.5 PER STATION
# =====================================
st.subheader("📊 Rata-rata PM2.5 per Station")

station_pm25 = df_filtered.groupby('station')['PM2.5'].mean().sort_values()

fig, ax = plt.subplots()
station_pm25.plot(kind='bar', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# =====================================
# 3. TREND WAKTU
# =====================================
st.subheader("📈 Trend PM2.5 Over Time")

time_series = df_filtered.groupby('datetime')['PM2.5'].mean()

fig, ax = plt.subplots()
time_series.plot(ax=ax)
st.pyplot(fig)

# =====================================
# 4. POLA BULANAN
# =====================================
st.subheader("📅 Pola Bulanan PM2.5")

monthly = df_filtered.groupby('month')['PM2.5'].mean()

fig, ax = plt.subplots()
monthly.plot(marker='o', ax=ax)
st.pyplot(fig)

# =====================================
# 5. POLA JAM
# =====================================
st.subheader("🕒 Pola PM2.5 per Jam")

hourly = df_filtered.groupby('hour')['PM2.5'].mean()

fig, ax = plt.subplots()
hourly.plot(ax=ax)
st.pyplot(fig)

# =====================================
# 6. HEATMAP
# =====================================
st.subheader("🔥 Heatmap PM2.5 (Bulan vs Jam)")

pivot = df_filtered.pivot_table(values='PM2.5', index='month', columns='hour')

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(pivot, ax=ax)
st.pyplot(fig)

# =====================================
# 7. KORELASI
# =====================================
st.subheader("🌪️ Korelasi Faktor Lingkungan")

corr = df_filtered[['PM2.5','TEMP','PRES','DEWP','RAIN','WSPM']].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
st.pyplot(fig)

# =====================================
# 8. CLUSTERING
# =====================================
st.subheader("🧠 Clustering Station")

from sklearn.cluster import KMeans

station_data = df_filtered.groupby('station')[['PM2.5','PM10','NO2']].mean()

kmeans = KMeans(n_clusters=3, random_state=42)
station_data['cluster'] = kmeans.fit_predict(station_data)

fig, ax = plt.subplots()
sns.scatterplot(data=station_data, x='PM2.5', y='PM10', hue='cluster', s=100, ax=ax)
st.pyplot(fig)

st.dataframe(station_data)

# =====================================
# 9. EXTREME POLLUTION
# =====================================
st.subheader("🚨 Extreme Pollution Detection")

threshold = df_filtered['PM2.5'].quantile(0.95)
extreme = df_filtered[df_filtered['PM2.5'] > threshold]

extreme_station = extreme['station'].value_counts()

fig, ax = plt.subplots()
extreme_station.plot(kind='bar', ax=ax)
st.pyplot(fig)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("Created by Raihan Okta Rahman")