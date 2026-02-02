import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="🎵 Spotify Cluster Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-size: 18px;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    .song-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .cluster-result {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stSelectbox label, .stMarkdown {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load the full dataset


@st.cache_data
def load_data():
    df = pd.read_csv('../spotify_tracks.csv')
    return df

# Load or train the model


@st.cache_resource
def load_model():
    try:
        kmeans_model = pickle.load(open('spotify_kmeans_model.pkl', 'rb'))
        scaler = pickle.load(open('spotify_scaler.pkl', 'rb'))
        st.success("✅ Loaded pre-trained model")
        return kmeans_model, scaler
    except:
        st.info("Training new model...")
        df = load_data()
        df_clustering = df.drop(
            columns=['track_name', 'artist', 'genre', 'playlist_category'])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clustering)

        kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans_model.fit(X_scaled)

        pickle.dump(kmeans_model, open('spotify_kmeans_model.pkl', 'wb'))
        pickle.dump(scaler, open('spotify_scaler.pkl', 'wb'))

        st.success("✅ Model trained and saved")
        return kmeans_model, scaler


# Cluster names and descriptions
cluster_names = {
    0: "Chill Danceable (Slow Tempo)",
    1: "Happy Long Tracks",
    2: "Sad Long Tracks",
    3: "Fast Danceable (High Tempo)",
    4: "Fast Short Tracks"
}

cluster_descriptions = {
    0: "Perfect for relaxed dancing with a slower beat. These tracks have high danceability but maintain a chill vibe.",
    1: "Uplifting and positive tracks with longer durations. Great for feel-good moments and extended listening sessions.",
    2: "Emotional and melancholic longer tracks. Perfect for introspective moments and deeper emotional experiences.",
    3: "High-energy danceable tracks with fast tempo. Ideal for workouts, parties, and energetic activities.",
    4: "Quick, upbeat tracks with high tempo. Great for quick energy bursts and dynamic playlists."
}


def convert_user_inputs_to_features(mood, activity, pace, length):
    """Convert user-friendly inputs to technical Spotify features"""

    mood_map = {
        'Happy & Upbeat': {'valence': 0.8, 'energy': 0.65},
        'Sad & Melancholic': {'valence': 0.2, 'energy': 0.35},
        'Energetic & Pumped': {'valence': 0.6, 'energy': 0.85},
        'Calm & Peaceful': {'valence': 0.5, 'energy': 0.25},
        'Angry & Intense': {'valence': 0.3, 'energy': 0.9}
    }

    activity_map = {
        'Party & Dancing': {'danceability': 0.8, 'energy': 0.85, 'tempo': 125, 'popularity': 70},
        'Workout & Exercise': {'danceability': 0.7, 'energy': 0.9, 'tempo': 140, 'popularity': 65},
        'Study & Focus': {'danceability': 0.3, 'energy': 0.3, 'tempo': 95, 'popularity': 50},
        'Sleep & Relaxation': {'danceability': 0.25, 'energy': 0.2, 'tempo': 75, 'popularity': 45},
        'Driving & Commute': {'danceability': 0.6, 'energy': 0.6, 'tempo': 115, 'popularity': 60},
        'Chill & Hangout': {'danceability': 0.5, 'energy': 0.4, 'tempo': 100, 'popularity': 55}
    }

    pace_map = {
        'Slow (Ballad)': {'tempo': 85},
        'Medium (Moderate)': {'tempo': 115},
        'Fast (Upbeat)': {'tempo': 150}
    }

    length_map = {
        'Short (< 3 min)': 150000,
        'Medium (3-4 min)': 220000,
        'Long (> 4 min)': 300000
    }

    features = {
        'danceability': 0.5,
        'energy': 0.5,
        'valence': 0.5,
        'tempo': 120,
        'duration_ms': 200000,
        'popularity': 50
    }

    if mood:
        features.update(mood_map.get(mood, {}))
    if activity:
        features.update(activity_map.get(activity, {}))
    if pace:
        features.update(pace_map.get(pace, {}))
    if length:
        features['duration_ms'] = length_map.get(length, 200000)

    return np.array([[
        features['danceability'],
        features['energy'],
        features['valence'],
        features['tempo'],
        features['duration_ms'],
        features['popularity']
    ]])

# Main App


def main():
    # Title
    st.markdown("<h1 style='text-align: center;'>🎵 Spotify Track Cluster Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Discover which music cluster your track belongs to using AI-powered K-means clustering</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Load data and model
    df = load_data()
    kmeans_model, scaler = load_model()

    # Assign clusters to all songs if not already done
    if 'cluster' not in df.columns:
        df_clustering = df.drop(
            columns=['track_name', 'artist', 'genre', 'playlist_category'])
        X_scaled = scaler.transform(df_clustering)
        df['cluster'] = kmeans_model.predict(X_scaled)

    # Sidebar for inputs
    st.sidebar.header("🎛️ Track Characteristics")

    mood = st.sidebar.selectbox(
        "😊 What's the mood?",
        ['', 'Happy & Upbeat', 'Sad & Melancholic',
            'Energetic & Pumped', 'Calm & Peaceful', 'Angry & Intense'],
        help="The overall emotional tone of the track"
    )

    activity = st.sidebar.selectbox(
        "🎯 What's it for?",
        ['', 'Party & Dancing', 'Workout & Exercise', 'Study & Focus',
            'Sleep & Relaxation', 'Driving & Commute', 'Chill & Hangout'],
        help="Best activity for this track"
    )

    pace = st.sidebar.selectbox(
        "⏱️ How fast is it?",
        ['', 'Slow (Ballad)', 'Medium (Moderate)', 'Fast (Upbeat)'],
        help="The speed/tempo of the track"
    )

    length = st.sidebar.selectbox(
        "⏳ How long is the track?",
        ['', 'Short (< 3 min)', 'Medium (3-4 min)', 'Long (> 4 min)'],
        help="Duration of the track"
    )

    predict_button = st.sidebar.button(
        "🎯 Predict Cluster", use_container_width=True)

    # Main content area
    if predict_button:
        if not all([mood, activity, pace, length]):
            st.error("⚠️ Please select all options before predicting!")
        else:
            with st.spinner("🔮 Analyzing your track..."):
                # Convert inputs to features
                features = convert_user_inputs_to_features(
                    mood, activity, pace, length)

                # Scale and predict
                features_scaled = scaler.transform(features)
                cluster = int(kmeans_model.predict(features_scaled)[0])

                # Get confidence
                distances = kmeans_model.transform(features_scaled)[0]
                confidence = (1 - (distances[cluster] / distances.sum())) * 100

                # Display results
                st.markdown("<div class='cluster-result'>",
                            unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(
                        f"<h2 style='color: #667eea;'>{cluster_names[cluster]}</h2>", unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='font-size: 16px; color: #555;'>{cluster_descriptions[cluster]}</p>", unsafe_allow_html=True)

                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.metric("Cluster ID", cluster)

                st.markdown("</div>", unsafe_allow_html=True)

                # Display input summary
                st.subheader("📊 Your Selection")
                cols = st.columns(4)
                with cols[0]:
                    st.info(f"**Mood**\n\n{mood}")
                with cols[1]:
                    st.info(f"**Activity**\n\n{activity}")
                with cols[2]:
                    st.info(f"**Pace**\n\n{pace}")
                with cols[3]:
                    st.info(f"**Length**\n\n{length}")

                # Display technical features
                st.subheader("🔬 Technical Features")
                feat_cols = st.columns(6)
                with feat_cols[0]:
                    st.metric("Danceability", f"{features[0][0]:.2f}")
                with feat_cols[1]:
                    st.metric("Energy", f"{features[0][1]:.2f}")
                with feat_cols[2]:
                    st.metric("Valence", f"{features[0][2]:.2f}")
                with feat_cols[3]:
                    st.metric("Tempo", f"{features[0][3]:.0f}")
                with feat_cols[4]:
                    st.metric("Duration", f"{features[0][4]/1000:.0f}s")
                with feat_cols[5]:
                    st.metric("Popularity", f"{features[0][5]:.0f}")

                # Display sample songs from the cluster
                st.subheader("🎵 Similar Songs in This Cluster")
                cluster_songs = df[df['cluster'] == cluster]
                sample_songs = cluster_songs.sample(min(3, len(cluster_songs)))

                for idx, song in sample_songs.iterrows():
                    st.markdown(f"""
                    <div class='song-card'>
                        <h4 style='color: #333; margin-bottom: 5px;'>🎵 {song['track_name']}</h4>
                        <p style='color: #666; margin: 0;'>👤 {song['artist']} <span style='background: #667eea; color: white; padding: 2px 10px; border-radius: 10px; font-size: 12px; margin-left: 10px;'>{song['genre']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.success("✅ Prediction Complete!")

    else:
        # Show welcome message
        st.info(
            "👈 Select track characteristics from the sidebar and click 'Predict Cluster' to get started!")

        # Show cluster statistics
        st.subheader("📈 Cluster Overview")
        if 'cluster' in df.columns:
            cluster_counts = df['cluster'].value_counts().sort_index()

            cols = st.columns(5)
            for i, col in enumerate(cols):
                count = cluster_counts.get(i, 0)
                with col:
                    st.metric(
                        cluster_names[i].split('(')[0].strip(),
                        f"{count} songs",
                        delta=f"{count/len(df)*100:.1f}%"
                    )


if __name__ == "__main__":
    main()
