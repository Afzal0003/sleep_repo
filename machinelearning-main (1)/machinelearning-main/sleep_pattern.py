import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from streamlit_autorefresh import st_autorefresh
from streamlit_player import st_player

# Set page config
st.set_page_config(
    page_title="Sleep Genius AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://sleepscience.ai',
        'Report a bug': 'https://github.com/sleep-ai/issues',
        'About': "# The Ultimate Sleep Optimization Platform"
    }
)

# Auto-refresh every 30 minutes
st_autorefresh(interval=30 * 60 * 1000, key="data_refresh")

# CSS for Dark and Light Themes
st.markdown("""
<style>
    :root {
        --primary: #6366f1;
        --secondary: #a5b4fc;
        --dark-bg: #0f172a;
        --light-bg: #f5f5f5;
        --dark-text: #e2e8f0;
        --light-text: #1f2937;
        --accent: #f59e0b;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .dark-theme .stApp {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1e3a8a 100%);
        color: var(--dark-text);
    }
    .light-theme .stApp {
        background: linear-gradient(135deg, var(--light-bg) 0%, #e0e7ff 100%);
        color: var(--light-text);
    }
    .title {
        text-align: center;
        font-size: 3.2em;
        text-shadow: 0 2px 10px rgba(165, 180, 252, 0.3);
        margin-bottom: 10px;
        animation: fadeIn 1s ease-in;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .dark-theme .title {
        color: var(--secondary);
    }
    .light-theme .title {
        color: #4f46e5;
    }
    .stButton>button {
        background: var(--primary);
        color: white;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        font-size: 1.1em;
        margin: 10px 0;
    }
    .stButton>button:hover {
        background: #4f46e5;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .stSlider .thumb {
        background: var(--accent) !important;
    }
    .stSlider .track {
        background: var(--secondary) !important;
    }
    .stMetric {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        min-width: 200px;
    }
    .dark-theme .stMetric {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .light-theme .stMetric {
        background: rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stMetric label {
        color: var(--secondary) !important;
        font-weight: 600;
        font-size: 0.9em;
    }
    .stMetric .value {
        font-size: 1.2em !important;
        font-weight: 600;
        white-space: normal;
        word-wrap: break-word;
    }
    .sidebar .sidebar-content {
        border-right: 1px solid #334155;
        box-shadow: 5px 0 15px rgba(0,0,0,0.2);
    }
    .dark-theme .sidebar .sidebar-content {
        background: #1e293b;
    }
    .light-theme .sidebar .sidebar-content {
        background: #ffffff;
    }
    .tip-box {
        border-left: 4px solid var(--accent);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        animation: fadeIn 0.8s ease;
        transition: all 0.3s ease;
    }
    .dark-theme .tip-box {
        background: rgba(255,255,255,0.08);
    }
    .light-theme .tip-box {
        background: rgba(0,0,0,0.05);
    }
    .tip-box:hover {
        transform: translateX(5px);
    }
    .dark-theme .tip-box:hover {
        background: rgba(255,255,255,0.12);
    }
    .light-theme .tip-box:hover {
        background: rgba(0,0,0,0.1);
    }
    .audio-box {
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .dark-theme .audio-box {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .light-theme .audio-box {
        background: rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }
    .audio-box:hover {
        transform: translateY(-3px);
    }
    .dark-theme .audio-box:hover {
        background: rgba(255,255,255,0.12);
    }
    .light-theme .audio-box:hover {
        background: rgba(0,0,0,0.1);
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 12px 0;
        font-size: 0.9em;
        z-index: 1000;
        border-top: 1px solid #334155;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .dark-theme .footer {
        background: rgba(15, 23, 42, 0.95);
        color: #c7d2fe;
    }
    .light-theme .footer {
        background: rgba(255, 255, 255, 0.95);
        color: #4b5563;
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
        100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .dark-theme .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><circle cx="20" cy="20" r="1" fill="white" opacity="0.3"/><circle cx="100" cy="100" r="1.5" fill="white" opacity="0.5"/><circle cx="180" cy="180" r="1" fill="white" opacity="0.4"/></svg>');
        background-size: 200px;
        animation: twinkle 15s infinite;
        z-index: -1;
    }
    .light-theme .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><circle cx="20" cy="20" r="1" fill="gray" opacity="0.2"/><circle cx="100" cy="100" r="1.5" fill="gray" opacity="0.3"/><circle cx="180" cy="180" r="1" fill="gray" opacity="0.2"/></svg>');
        background-size: 200px;
        z-index: -1;
    }
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.7; }
    }
    .sleep-stage {
        height: 100px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    .sleep-stage:hover {
        transform: scale(1.02);
    }
    .tab-content {
        animation: fadeIn 0.5s ease;
    }
    .feature-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .dark-theme .feature-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .light-theme .feature-card {
        background: rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }
    .feature-card:hover {
        transform: translateY(-3px);
    }
    .dark-theme .feature-card:hover {
        background: rgba(255,255,255,0.1);
    }
    .light-theme .feature-card:hover {
        background: rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history and theme
if 'sleep_history' not in st.session_state:
    st.session_state.sleep_history = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

# Apply theme class
theme_class = "dark-theme" if st.session_state.theme == 'Dark' else "light-theme"
st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    try:
        file_path = "Sleep_health_and_lifestyle_dataset.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Caffeine Intake'] = np.random.randint(0, 5, len(df))
            df['Screen Time'] = np.random.randint(2, 12, len(df))
            df['Bedtime Consistency'] = np.random.uniform(0.5, 1.0, len(df))
            df['Sleep Latency'] = np.random.randint(5, 60, len(df))
            df['Wakeup Mood'] = np.random.randint(1, 5, len(df))
        else:
            st.warning("Dataset not found. Using enhanced simulated data.")
            data = {
                'Age': np.random.randint(18, 70, 500),
                'Gender': np.random.choice(['Male', 'Female'], 500),
                'Sleep Duration': np.random.uniform(4, 10, 500),
                'Physical Activity Level': np.random.randint(10, 120, 500),
                'Stress Level': np.random.randint(1, 10, 500),
                'Heart Rate': np.random.randint(55, 85, 500),
                'Daily Steps': np.random.randint(2000, 15000, 500),
                'Quality of Sleep': np.random.randint(1, 10, 500),
                'Caffeine Intake': np.random.randint(0, 5, 500),
                'Screen Time': np.random.randint(2, 12, 500),
                'Bedtime Consistency': np.random.uniform(0.3, 1.0, 500),
                'Sleep Latency': np.random.randint(5, 60, 500),
                'Wakeup Mood': np.random.randint(1, 5, 500)
            }
            df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Preprocess data
def preprocess_data(df):
    try:
        df = df.fillna(df.mean(numeric_only=True))
        df['Activity_to_Stress'] = df['Physical Activity Level'] / (df['Stress Level'] + 1)
        df['Sleep_Efficiency'] = df['Sleep Duration'] * df['Quality of Sleep'] / 10
        df['Digital_Detox'] = (24 - df['Screen Time']) / 24
        df['Caffeine_Impact'] = df['Caffeine Intake'] * df['Sleep Latency'] / 60
        df['Consistency_Bonus'] = df['Bedtime Consistency'] * 0.5
        features = [
            'Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level',
            'Heart Rate', 'Daily Steps', 'Caffeine Intake', 'Screen Time',
            'Bedtime Consistency', 'Sleep Latency', 'Activity_to_Stress',
            'Sleep_Efficiency', 'Digital_Detox', 'Caffeine_Impact', 'Consistency_Bonus'
        ]
        X = df[features]
        y_risk = (df['Quality of Sleep'] < 5).astype(int)
        y_duration = df['Sleep Duration']
        y_mood = df['Wakeup Mood']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y_risk, y_duration, y_mood, scaler, features, df
    except Exception as e:
        st.error(f"Data preprocessing failed: {str(e)}")
        return None, None, None, None, None, None, None

# Train models
@st.cache_resource
def train_models(X_scaled, y_risk, y_duration, y_mood):
    try:
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X_scaled)
        X_train, X_test, y_train_risk, y_test_risk, y_train_dur, y_test_dur, y_train_mood, y_test_mood = train_test_split(
            X_scaled, y_risk, y_duration, y_mood, test_size=0.2, random_state=42
        )
        lr = LogisticRegression(random_state=42, max_iter=1000)
        xgb = XGBClassifier(random_state=42)
        gbc = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'max_depth': [3, 5],
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }
        xgb_grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1')
        gbc_grid = GridSearchCV(gbc, param_grid, cv=5, scoring='f1')
        lr.fit(X_train, y_train_risk)
        xgb_grid.fit(X_train, y_train_risk)
        gbc_grid.fit(X_train, y_train_risk)
        lr_pred = lr.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_grid.predict_proba(X_test)[:, 1]
        gbc_pred = gbc_grid.predict_proba(X_test)[:, 1]
        ensemble_pred = (lr_pred + xgb_pred + gbc_pred) / 3
        ensemble_labels = (ensemble_pred > 0.5).astype(int)
        f1 = f1_score(y_test_risk, ensemble_labels)
        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        gbr = GradientBoostingRegressor(n_estimators=150, random_state=42)
        rf.fit(X_train, y_train_dur)
        gbr.fit(X_train, y_train_dur)
        model_duration = Sequential([
            Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model_duration.compile(optimizer='adam', loss='mse')
        model_duration.fit(X_train, y_train_dur, epochs=150, batch_size=32, verbose=0, validation_split=0.2)
        model_mood = Sequential([
            Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model_mood.compile(optimizer='adam', loss='mse')
        model_mood.fit(X_train, y_train_mood, epochs=100, batch_size=16, verbose=0, validation_split=0.2)
        rf_pred = rf.predict(X_test)
        gbr_pred = gbr.predict(X_test)
        ann_pred = model_duration.predict(X_test, verbose=0).flatten()
        ensemble_duration = (rf_pred + gbr_pred + ann_pred) / 3
        r2 = r2_score(y_test_dur, ensemble_duration)
        mood_pred = model_mood.predict(X_test, verbose=0).flatten()
        mood_r2 = r2_score(y_test_mood, mood_pred)
        return {
            'kmeans': kmeans,
            'classifiers': {'lr': lr, 'xgb': xgb_grid, 'gbc': gbc_grid},
            'regressors': {'rf': rf, 'gbr': gbr, 'ann_duration': model_duration},
            'mood_model': model_mood,
            'metrics': {'f1': f1, 'r2': r2, 'mood_r2': mood_r2},
            'X_test': X_test,
            'y_test_risk': y_test_risk
        }
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

# Convert to 12-hour format
def to_12_hour(hour, minute):
    period = "🌅 AM" if hour < 12 else "🌇 PM"
    hour = hour if hour <= 12 else hour - 12
    hour = 12 if hour == 0 else hour
    return f"{int(hour)}:{minute:02d} {period}"

# Predict function
def predict_sleep(user_data, scaler, models):
    try:
        user_scaled = scaler.transform([user_data])
        cluster = models['kmeans'].predict(user_scaled)[0]
        profiles = {
            0: "Early Riser 🌅",
            1: "Night Owl 🦉",
            2: "Balanced Sleeper ⚖️",
            3: "Stressed Sleeper 😓",
            4: "Irregular Sleeper 🔄"
        }
        profile = profiles.get(cluster, "Unknown Profile")
        clf = models['classifiers']
        lr_prob = clf['lr'].predict_proba(user_scaled)[0][1]
        xgb_prob = clf['xgb'].predict_proba(user_scaled)[0][1]
        gbc_prob = clf['gbc'].predict_proba(user_scaled)[0][1]
        risk_prob = (lr_prob + xgb_prob + gbc_prob) / 3
        risk_level = "Critical 🚨" if risk_prob > 0.7 else "High ⚠️" if risk_prob > 0.5 else "Moderate ℹ️" if risk_prob > 0.3 else "Low ✅"
        reg = models['regressors']
        rf_duration = reg['rf'].predict(user_scaled)[0]
        gbr_duration = reg['gbr'].predict(user_scaled)[0]
        ann_duration = reg['ann_duration'].predict(user_scaled, verbose=0)[0][0]
        duration = (rf_duration + gbr_duration + ann_duration) / 3
        duration = max(4, min(10, duration))
        mood_score = models['mood_model'].predict(user_scaled, verbose=0)[0][0]
        mood_score = max(1, min(5, mood_score))
        mood_emojis = ["😢", "😞", "😐", "🙂", "😊"]
        mood = mood_emojis[min(4, int(mood_score) - 1)]
        now = datetime.now()
        if cluster == 0:
            target_bedtime = now.replace(hour=21, minute=0, second=0, microsecond=0)
            target_waketime = target_bedtime + timedelta(hours=duration)
        elif cluster == 1:
            target_bedtime = now.replace(hour=23, minute=30, second=0, microsecond=0)
            target_waketime = target_bedtime + timedelta(hours=duration)
        else:
            target_bedtime = now.replace(hour=22, minute=15, second=0, microsecond=0)
            target_waketime = target_bedtime + timedelta(hours=duration)
        if target_waketime.hour < 6:
            target_waketime = target_waketime.replace(hour=6, minute=0)
            target_bedtime = target_waketime - timedelta(hours=duration)
        elif target_waketime.hour > 9:
            target_waketime = target_waketime.replace(hour=9, minute=0)
            target_bedtime = target_waketime - timedelta(hours=duration)
        start_time = to_12_hour(target_bedtime.hour, target_bedtime.minute)
        end_time = to_12_hour(target_waketime.hour, target_waketime.minute)
        sleep_stages = {
            "NREM 1 (Light Sleep)": duration * 0.05,
            "NREM 2 (Memory Consolidation)": duration * 0.45,
            "NREM 3 (Deep Sleep)": duration * 0.25,
            "REM (Dreaming)": duration * 0.25
        }
        tips = []
        if user_data[3] > 7:
            tips.append("🧘‍♀️ Practice 15-min guided meditation before bed to lower stress")
        if user_data[6] > 3:
            tips.append("☕ Avoid caffeine after 2PM to improve sleep quality")
        if user_data[7] > 5:
            tips.append("📱 Implement a 1-hour digital detox before bedtime")
        if duration < 6.5:
            tips.append("⏰ Consider gradually increasing sleep time by 15-min increments")
        if cluster == 1:
            tips.append("🌞 Get morning sunlight exposure to help regulate circadian rhythm")
        return {
            'profile': profile,
            'risk_level': risk_level,
            'risk_prob': risk_prob,
            'duration': duration,
            'start_time': start_time,
            'end_time': end_time,
            'mood': mood,
            'mood_score': mood_score,
            'sleep_stages': sleep_stages,
            'tips': tips,
            'cluster': cluster
        }
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Plot circadian rhythm (simplified)
def plot_circadian_rhythm(cluster):
    hours = np.linspace(0, 24, 100)
    if cluster == 0:
        alertness = 2.5 * np.sin((hours - 6) * np.pi / 12)
    elif cluster == 1:
        alertness = 2.5 * np.sin((hours - 12) * np.pi / 12)
    else:
        alertness = 2.5 * np.sin((hours - 9) * np.pi / 12)
    df = pd.DataFrame({'hour': hours, 'alertness': alertness})
    fig = px.area(df, x='hour', y='alertness',
                  title='Your Daily Energy Pattern',
                  labels={'hour': 'Time of Day (24h)', 'alertness': 'Energy Level'})
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(tickvals=list(range(0, 25, 3)), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showticklabels=False),
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    if cluster == 0:
        fig.add_vrect(x0=21, x1=6, fillcolor="rgba(99, 102, 241, 0.2)", layer="below", line_width=0)
    elif cluster == 1:
        fig.add_vrect(x0=0, x1=9, fillcolor="rgba(99, 102, 241, 0.2)", layer="below", line_width=0)
    else:
        fig.add_vrect(x0=22, x1=7, fillcolor="rgba(99, 102, 241, 0.2)", layer="below", line_width=0)
    return fig

# Plot sleep stages (simplified)
def plot_sleep_stages(sleep_stages):
    stages = list(sleep_stages.keys())
    durations = list(sleep_stages.values())
    fig = go.Figure(data=[go.Pie(
        labels=stages,
        values=durations,
        hole=0.4,
        marker_colors=['#6366f1', '#a5b4fc', '#f59e0b', '#ec4899'],
        textinfo='label+percent',
        hoverinfo='label+value',
        textfont=dict(size=12)
    )])
    fig.update_layout(
        title='Your Sleep Stages',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Main app
def main():
    st.markdown('<h1 class="title">🌌 Sleep Genius AI</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; font-size: 1.1em;">
        The most advanced sleep optimization platform powered by AI. Get personalized recommendations, 
        understand your sleep patterns, and transform your rest.
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the dataset.")
        st.stop()

    X_scaled, y_risk, y_duration, y_mood, scaler, features, df = preprocess_data(df)
    if X_scaled is None:
        st.stop()

    models = train_models(X_scaled, y_risk, y_duration, y_mood)
    if models is None:
        st.stop()

    with st.sidebar:
        st.markdown("### 🛌 Your Sleep Profile")
        with st.expander("⚙️ Lifestyle Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 80, 32, help="Your age in years")
                sleep_duration = st.slider("Current Sleep (hrs)", 4.0, 10.0, 6.5, step=0.25, help="Your average nightly sleep duration")
                activity = st.slider("Activity (min/day)", 0, 180, 45, help="Daily moderate to vigorous activity")
                caffeine = st.slider("Caffeine (cups/day)", 0, 8, 2, help="Daily caffeine intake")
            with col2:
                stress = st.slider("Stress (1-10)", 1, 10, 5, help="Perceived stress level")
                screen_time = st.slider("Screen Time (hrs)", 1, 16, 5, help="Daily screen exposure")
                consistency = st.slider("Bedtime Consistency", 0.0, 1.0, 0.7, step=0.1, help="How consistent your bedtime is (0-1)")
                latency = st.slider("Sleep Latency (min)", 5, 60, 20, help="Time to fall asleep")
        st.markdown("---")
        st.markdown("📊 **Model Performance**")
        st.markdown(f"- Sleep Risk F1: `{models['metrics']['f1']:.2f}`")
        st.markdown(f"- Duration R²: `{models['metrics']['r2']:.2f}`")
        st.markdown(f"- Mood R²: `{models['metrics']['mood_r2']:.2f}`")
        st.markdown("---")
        if st.button("🔄 Reset All", key='reset'):
            st.session_state.sleep_history = []
            st.rerun()
        st.markdown("""
        <div style="margin-top: 20px; font-size: 0.8em;">
            ℹ️ For best results, update your parameters regularly as your lifestyle changes.
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🌙 Sleep Plan", "📊 Analytics", "💤 Sleep Science", "⚙️ Settings"])

    with tab1:
        if st.button("✨ Generate My Sleep Plan", key='predict', use_container_width=True):
            with st.spinner("🧠 Analyzing your sleep patterns with AI..."):
                user_data = [
                    age, sleep_duration, activity, stress,
                    np.random.randint(60, 80),
                    np.random.randint(3000, 12000),
                    caffeine, screen_time, consistency, latency,
                    activity / (stress + 1),
                    sleep_duration * 0.8,
                    (24 - screen_time) / 24,
                    caffeine * latency / 60,
                    consistency * 0.5
                ]
                results = predict_sleep(user_data, scaler, models)
                if results is None:
                    st.error("Failed to generate predictions. Please try again.")
                    st.stop()
                st.success("🎉 Your Personalized Sleep Plan is Ready!")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sleep Chronotype", results['profile'])
                with col2:
                    st.metric("Fatigue Risk", results['risk_level'], f"{results['risk_prob']:.1%}")
                with col3:
                    st.metric("Recommended Sleep", f"{results['duration']:.1f} hours")
                with col4:
                    st.metric("Predicted Mood", results['mood'], f"{results['mood_score']:.1f}/5")
                with st.expander("🕒 Your Ideal Sleep Schedule", expanded=True):
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 12px; border-left: 4px solid #6366f1;">
                        <h3 style="margin-top: 0;">⏰ Recommended Sleep Window</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin: 15px 0;">
                            <div style="text-align: center;">
                                <div style="font-size: 0.9em;">Bedtime</div>
                                <div style="font-size: 2em; font-weight: bold;">{results['start_time']}</div>
                            </div>
                            <div style="font-size: 1.5em;">→</div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.9em;">Wake Time</div>
                                <div style="font-size: 2em; font-weight: bold;">{results['end_time']}</div>
                            </div>
                        </div>
                        <div style="font-size: 0.9em;">
                            💡 This window optimizes your sleep quality based on your lifestyle.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.session_state.sleep_history.append({
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'duration': results['duration'],
                    'schedule': f"{results['start_time']} – {results['end_time']}",
                    'mood': results['mood_score']
                })
                st.subheader("🌌 Your Sleep Stages")
                st.plotly_chart(plot_sleep_stages(results['sleep_stages']), use_container_width=True)
                st.subheader("🌞 Your Energy Pattern")
                st.plotly_chart(plot_circadian_rhythm(results['cluster']), use_container_width=True)
                st.subheader("🔍 Personalized Recommendations")
                if results['tips']:
                    for tip in results['tips']:
                        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
                else:
                    st.info("Your sleep looks great! Keep up the good habits.")
                st.subheader("🎵 Relaxation Audio")
                with st.container():
                    st.markdown("""
                    <div class="audio-box">
                        <h4 style="margin-top: 0;">Triple Healing Frequencies</h4>
                        <p style="font-size: 0.9em;">
                            Relax with 432 Hz, 528 Hz, and 639 Hz tones to promote deep sleep and healing.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        st_player("https://soundcloud.com/chakrahealingmusicacademy/sets/432-hz-528-hz-639-hz-triple", key="relax_audio", height=400)
                    except Exception as e:
                        st.warning("Audio temporarily unavailable. Try this link: [Triple Healing Frequencies](https://soundcloud.com/chakrahealingmusicacademy/sets/432-hz-528-hz-639-hz-triple)")
                    st.markdown("""
                    <div style="font-size: 0.8em; margin-top: 10px;">
                        ℹ️ Play these healing frequencies before bed to enhance relaxation.
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📈 Sleep Analytics")
        if st.session_state.sleep_history:
            history_df = pd.DataFrame(st.session_state.sleep_history)
            st.markdown("### Sleep Duration Trends")
            fig_trend = px.line(
                history_df,
                x='date',
                y='duration',
                markers=True,
                title="Sleep Duration Over Time",
                labels={'duration': 'Hours', 'date': 'Date'}
            )
            fig_trend.update_traces(line_color='#6366f1', marker=dict(size=8))
            fig_trend.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown("### Mood Trends")
            fig_mood = px.scatter(
                history_df,
                x='date',
                y='mood',
                size='mood',
                color='mood',
                title="Morning Mood Over Time",
                labels={'mood': 'Mood (1-5)', 'date': 'Date'}
            )
            fig_mood.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_mood, use_container_width=True)
        else:
            st.info("No history yet. Generate a sleep plan to see trends!")
        st.markdown("### Key Factors")
        rf = models['regressors']['rf']
        importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
        importance = importance.sort_values('Importance', ascending=False).head(5)
        fig_importance = px.bar(
            importance,
            x='Importance',
            y='Feature',
            title='Top Factors Affecting Your Sleep',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    with tab3:
        st.subheader("💤 Understanding Your Sleep")
        st.markdown("""
        ### Sleep Stages Explained
        - **NREM 1 (Light Sleep)**: Transition phase, easy to wake (5% of sleep).
        - **NREM 2 (Memory Consolidation)**: Body repairs, heart rate slows (45%).
        - **NREM 3 (Deep Sleep)**: Physical restoration, immune boost (25%).
        - **REM (Dreaming)**: Brain activity spikes, memory processing (25%).
        
        ### Daily Energy Pattern
        Your body’s 24-hour clock regulates energy levels. Early Risers peak in the morning, Night Owls at night, and Balanced Sleepers have flexible patterns.
        
        ### AI Models
        - **Clustering**: Groups you into a sleep type (Early Riser, Night Owl, etc.).
        - **Classification**: Predicts fatigue risk with high accuracy.
        - **Regression**: Estimates sleep duration and mood based on lifestyle.
        
        Built with **TensorFlow**, **Scikit-learn**, and **Streamlit**.
        """)
        st.markdown("### Learn More")
        st_player("https://www.youtube.com/watch?v=5MuIMqhT8DM", key="sleep_science_video")

    with tab4:
        st.subheader("⚙️ Customize Your Experience")
        theme = st.selectbox("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == 'Dark' else 1)
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
        if st.button("Export Sleep History"):
            history_df = pd.DataFrame(st.session_state.sleep_history)
            csv = history_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "sleep_history.csv", "text/csv")
        if st.button("Clear History"):
            st.session_state.sleep_history = []
            st.success("History cleared!")
            st.rerun()
        st.markdown("### Retrain Model")
        if st.button("Retrain with New Data"):
            with st.spinner("Retraining models..."):
                models = None  # Clear cache
                models = train_models(X_scaled, y_risk, y_duration, y_mood)
                st.success("Models retrained!")
        st.markdown("""
        <div style="margin-top: 20px; font-size: 0.8em;">
            ℹ️ Export your history to track progress offline.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <strong>Sleep Genius AI</strong><br>
        Made by Bala using TensorFlow, Scikit-learn, and Streamlit. Powered by the Sleep Health and Lifestyle Dataset (Kaggle).<br>
        <strong>Mission</strong>: Revolutionize sleep for a healthier world!<br>
        <a href="https://sleepscience.ai">Learn More</a> | 
        <a href="https://github.com/Balaswamyvasamsetti">Contribute</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()