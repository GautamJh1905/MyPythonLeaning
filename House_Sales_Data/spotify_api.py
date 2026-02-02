from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load the full dataset with song names and genres
df_full = pd.read_csv('../spotify_tracks.csv')

# Load or train the model
try:
    # Try to load pre-trained model
    kmeans_model = pickle.load(open('spotify_kmeans_model.pkl', 'rb'))
    scaler = pickle.load(open('spotify_scaler.pkl', 'rb'))
    # Load cluster assignments
    df_full['cluster'] = pickle.load(open('spotify_clusters.pkl', 'rb'))
    print("Loaded pre-trained model")
except:
    # If not available, train a new model
    print("Training new model...")
    df_clustering = df_full.drop(
        columns=['track_name', 'artist', 'genre', 'playlist_category'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clustering)

    kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)

    # Assign clusters to all songs
    df_full['cluster'] = kmeans_model.predict(X_scaled)

    # Save the model and clusters
    pickle.dump(kmeans_model, open('spotify_kmeans_model.pkl', 'wb'))
    pickle.dump(scaler, open('spotify_scaler.pkl', 'wb'))
    pickle.dump(df_full['cluster'].values, open('spotify_clusters.pkl', 'wb'))
    print("Model trained and saved")

# Cluster names mapping
cluster_names = {
    0: "Chill Danceable (Slow Tempo)",
    1: "Happy Long Tracks",
    2: "Sad Long Tracks",
    3: "Fast Danceable (High Tempo)",
    4: "Fast Short Tracks"
}

# Cluster descriptions
cluster_descriptions = {
    0: "Perfect for relaxed dancing with a slower beat. These tracks have high danceability but maintain a chill vibe.",
    1: "Uplifting and positive tracks with longer durations. Great for feel-good moments and extended listening sessions.",
    2: "Emotional and melancholic longer tracks. Perfect for introspective moments and deeper emotional experiences.",
    3: "High-energy danceable tracks with fast tempo. Ideal for workouts, parties, and energetic activities.",
    4: "Quick, upbeat tracks with high tempo. Great for quick energy bursts and dynamic playlists."
}


def convert_user_inputs_to_features(data):
    """Convert user-friendly inputs to technical Spotify features"""

    # Mood mapping
    mood_map = {
        'happy': {'valence': 0.8, 'energy': 0.65},
        'sad': {'valence': 0.2, 'energy': 0.35},
        'energetic': {'valence': 0.6, 'energy': 0.85},
        'calm': {'valence': 0.5, 'energy': 0.25},
        'angry': {'valence': 0.3, 'energy': 0.9}
    }

    # Activity mapping
    activity_map = {
        'party': {'danceability': 0.8, 'energy': 0.85, 'tempo': 125, 'popularity': 70},
        'workout': {'danceability': 0.7, 'energy': 0.9, 'tempo': 140, 'popularity': 65},
        'study': {'danceability': 0.3, 'energy': 0.3, 'tempo': 95, 'popularity': 50},
        'sleep': {'danceability': 0.25, 'energy': 0.2, 'tempo': 75, 'popularity': 45},
        'drive': {'danceability': 0.6, 'energy': 0.6, 'tempo': 115, 'popularity': 60},
        'chill': {'danceability': 0.5, 'energy': 0.4, 'tempo': 100, 'popularity': 55}
    }

    # Pace mapping
    pace_map = {
        'fast': {'tempo': 150},
        'medium': {'tempo': 115},
        'slow': {'tempo': 85}
    }

    # Length mapping (in milliseconds)
    length_map = {
        'short': 150000,   # 2.5 minutes
        'medium': 220000,  # 3.7 minutes
        'long': 300000     # 5 minutes
    }

    # Start with defaults
    features = {
        'danceability': 0.5,
        'energy': 0.5,
        'valence': 0.5,
        'tempo': 120,
        'duration_ms': 200000,
        'popularity': 50
    }

    # Apply mood
    if 'mood' in data:
        mood_features = mood_map.get(data['mood'], {})
        features.update(mood_features)

    # Apply activity
    if 'activity' in data:
        activity_features = activity_map.get(data['activity'], {})
        features.update(activity_features)

    # Apply pace (override tempo if specified)
    if 'pace' in data:
        pace_features = pace_map.get(data['pace'], {})
        features.update(pace_features)

    # Apply length
    if 'length' in data:
        features['duration_ms'] = length_map.get(data['length'], 200000)

    return np.array([[
        features['danceability'],
        features['energy'],
        features['valence'],
        features['tempo'],
        features['duration_ms'],
        features['popularity']
    ]])


@app.route('/')
def home():
    return render_template('spotify_index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Check if user-friendly inputs or technical inputs
        if 'mood' in data:
            # Convert user-friendly inputs to technical features
            features = convert_user_inputs_to_features(data)
        else:
            # Extract features in the correct order (technical mode)
            features = np.array([[
                float(data['danceability']),
                float(data['energy']),
                float(data['valence']),
                float(data['tempo']),
                float(data['duration_ms']),
                float(data['popularity'])
            ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict cluster
        cluster = int(kmeans_model.predict(features_scaled)[0])
        cluster_name = cluster_names[cluster]
        description = cluster_descriptions[cluster]

        # Get distances to all centroids for confidence
        distances = kmeans_model.transform(features_scaled)[0]
        confidence = 1 - (distances[cluster] / distances.sum())

        # Get 3 sample songs from the same cluster
        cluster_songs = df_full[df_full['cluster'] == cluster]
        sample_songs = cluster_songs.sample(min(3, len(cluster_songs)))

        songs_list = []
        for _, song in sample_songs.iterrows():
            songs_list.append({
                'track_name': song['track_name'],
                'artist': song['artist'],
                'genre': song['genre']
            })

        return jsonify({
            'success': True,
            'cluster': cluster,
            'cluster_name': cluster_name,
            'description': description,
            'confidence': round(confidence * 100, 2),
            'sample_songs': songs_list,
            'features': {
                'danceability': float(features[0][0]),
                'energy': float(features[0][1]),
                'valence': float(features[0][2]),
                'tempo': float(features[0][3]),
                'duration_ms': float(features[0][4]),
                'popularity': float(features[0][5])
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/cluster_info', methods=['GET'])
def cluster_info():
    """Get information about all clusters"""
    centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)

    cluster_data = []
    for i in range(5):
        cluster_data.append({
            'cluster': i,
            'name': cluster_names[i],
            'description': cluster_descriptions[i],
            'centroid': {
                'danceability': round(centroids[i][0], 2),
                'energy': round(centroids[i][1], 2),
                'valence': round(centroids[i][2], 2),
                'tempo': round(centroids[i][3], 2),
                'duration_ms': round(centroids[i][4], 2),
                'popularity': round(centroids[i][5], 2)
            }
        })

    return jsonify({
        'success': True,
        'clusters': cluster_data
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
