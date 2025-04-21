import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
import math

# --- Constants ---
DATA_FILE_PATH = 'raw_data/loc-gowalla_totalCheckins.txt'
COLUMN_NAMES = ['user_id', 'timestamp', 'latitude', 'longitude', 'location_id']
OUTPUT_DIR = 'output_plots'
SAMPLE_SIZE = 1_000_000  # Using 1 million rows sample
MIN_USER_CHECKINS = 5
MIN_LOCATION_CHECKINS = 5
RECOMMENDATION_K = 10
TOP_K_NEIGHBORS_USER = 50 # Number of neighbors for UserBasedCF
GEO_WEIGHT_ALPHA = 0.3 # Weight for geographical influence in GeoUserBasedCF (0=cosine only, 1=geo only)

# --- Helper Functions ---
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# --- Data Loading & Preprocessing ---
def load_and_preprocess_data(file_path, nrows=None, min_user_checkins=5, min_loc_checkins=5):
    """Loads, samples, filters, and preprocesses the Gowalla check-in data."""
    print(f"Loading data from {file_path}...{' (Sample: ' + str(nrows) + ' rows)' if nrows else ''}")
    start_time = time.time()
    try:
        dtypes = {'user_id': int, 'timestamp': str, 'latitude': float, 'longitude': float, 'location_id': int}
        df = pd.read_csv(file_path, sep='\t', header=None, names=COLUMN_NAMES, dtype=dtypes, nrows=nrows)
        print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds. Initial rows: {len(df)}")

        # --- Filtering ---
        print(f"Filtering data (min_user={min_user_checkins}, min_loc={min_loc_checkins})...")
        while True:
            initial_rows = len(df)
            user_counts = df.groupby('user_id').size()
            df = df[df['user_id'].isin(user_counts[user_counts >= min_user_checkins].index)]

            location_counts = df.groupby('location_id').size()
            df = df[df['location_id'].isin(location_counts[location_counts >= min_loc_checkins].index)]

            # Repeat filtering if rows were removed
            if len(df) == initial_rows:
                break
            print(f" Filtering iteration removed rows, {len(df)} remaining...")

        print(f"Filtering complete. Final rows: {len(df)}, Unique Users: {df['user_id'].nunique()}, Unique Locations: {df['location_id'].nunique()}")

        # --- Feature Engineering (Integer IDs for Matrix) ---
        user_encoder = LabelEncoder()
        location_encoder = LabelEncoder()
        df['user_idx'] = user_encoder.fit_transform(df['user_id'])
        df['location_idx'] = location_encoder.fit_transform(df['location_id'])

        print(f"Preprocessing finished in {time.time() - start_time:.2f} seconds.")
        return df, user_encoder, location_encoder

    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error during data loading/preprocessing: {e}")
        return None, None, None

# --- EDA Functions ---
def plot_distribution(data, column_name, title, filename, output_dir, log_scale=True):
    """Plots and saves the distribution of counts for a given column."""
    if data is None:
        print(f"Data is None, cannot plot {title}.")
        return
    print(f"Plotting {title}...")
    start_time = time.time()
    counts = data.groupby(column_name).size()

    plt.figure(figsize=(12, 6))
    sns.histplot(counts, bins=50, kde=False)
    plt.title(title)
    plt.xlabel(f'Number of Check-ins')
    plt.ylabel(f'Number of {column_name.split("_")[0].capitalize()}s')
    if log_scale:
        plt.yscale('log')
    plt.grid(axis='y', alpha=0.5)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(plot_path)
        print(f" Plot saved to: {plot_path}")
    except Exception as e:
        print(f" Error saving plot: {e}")
    plt.close()
    print(f" Plotting completed in {time.time() - start_time:.2f} seconds.")

def plot_checkin_heatmap(df, output_dir):
    print("Plotting Check-in Heatmap...")
    start_time = time.time()
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    # Reproject to a suitable CRS for visualization (e.g., Web Mercator for contextily)
    gdf_wm = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Use seaborn's kdeplot for heatmap effect
    try:
        # Reduce sample size for KDE plot if dataframe is very large
        sample_size = min(len(gdf_wm), 50000) 
        gdf_sample = gdf_wm.sample(sample_size) if len(gdf_wm) > sample_size else gdf_wm
        sns.kdeplot(
            x=gdf_sample.geometry.x,
            y=gdf_sample.geometry.y,
            cmap="viridis", # Changed color map
            fill=True, 
            bw_adjust=.5,
            ax=ax,
            alpha=0.6 # Added transparency
        )
        # Add basemap
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        ax.set_axis_off()
        plt.title('Geographical Heatmap of Check-ins (Training Data)', fontsize=16)
        plot_path = os.path.join(output_dir, 'checkin_heatmap.png')
        plt.savefig(plot_path, bbox_inches='tight')
        print(f" Heatmap plot saved to: {plot_path}")
    except Exception as e:
        print(f" Could not generate heatmap plot: {e}")
    finally:
        plt.close(fig)
    print(f"Heatmap plotting finished in {time.time() - start_time:.2f} seconds.")

def plot_user_footprint(df, user_id_to_plot, user_encoder, output_dir):
    print(f"Plotting Footprint for User ID: {user_id_to_plot}...")
    start_time = time.time()
    try:
        user_idx_to_plot = user_encoder.transform([user_id_to_plot])[0]
    except ValueError:
        print(f" User ID {user_id_to_plot} not found in the filtered dataset.")
        return

    user_df = df[df['user_idx'] == user_idx_to_plot]
    if user_df.empty:
        print(f" No check-ins found for User ID {user_id_to_plot} in the training data.")
        return
    
    gdf_user = gpd.GeoDataFrame(user_df, geometry=gpd.points_from_xy(user_df.longitude, user_df.latitude), crs="EPSG:4326")
    gdf_user_wm = gdf_user.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf_user_wm.plot(ax=ax, marker='o', color='red', markersize=15, alpha=0.7, label=f'User {user_id_to_plot} Check-ins')
    
    # Calculate extent with padding
    minx, miny, maxx, maxy = gdf_user_wm.total_bounds
    padding = max((maxx - minx), (maxy-miny)) * 0.2 # Add 20% padding based on larger dimension
    ax.set_xlim(minx - padding, maxx + padding)
    ax.set_ylim(miny - padding, maxy + padding)
    
    try:
      cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
      ax.set_axis_off()
      plt.title(f'Geographical Footprint of User {user_id_to_plot} (Training Check-ins)', fontsize=14)
      plot_path = os.path.join(output_dir, f'user_{user_id_to_plot}_footprint.png')
      plt.savefig(plot_path, bbox_inches='tight')
      print(f" User footprint plot saved to: {plot_path}")
    except Exception as e:
        print(f" Could not add basemap or save user footprint plot: {e}")
    finally:
      plt.close(fig)
    print(f"Footprint plotting finished in {time.time() - start_time:.2f} seconds.")

# --- Train-Test Split ---
def temporal_train_test_split(df):
    """Splits data temporally: last check-in per user is test, rest is train."""
    print("Performing temporal train-test split...")
    start_time = time.time()
    # Assuming the last entry for a user in the sample is their 'latest'
    df_sorted = df.sort_values(by=['user_idx', 'timestamp']).copy() # Keep timestamp for potential future use
    test_indices = df_sorted.groupby('user_idx').tail(1).index
    train_df = df.drop(test_indices)
    test_df = df.loc[test_indices]
    print(f"Split complete: Train size={len(train_df)}, Test size={len(test_df)} in {time.time() - start_time:.2f} seconds.")
    # Create sets for quick lookup during evaluation
    test_user_item_set = set(zip(test_df['user_idx'], test_df['location_idx']))
    # Get all items user interacted with in training for filtering recommendations
    train_user_items_map = train_df.groupby('user_idx')['location_idx'].agg(set).to_dict()
    return train_df, test_df, test_user_item_set, train_user_items_map

# --- Recommendation Models ---
class MostFrequentRecommender:
    def __init__(self):
        self.location_popularity = None
        self.train_user_items_map = None

    def fit(self, train_df, train_user_items_map):
        print("Fitting MostFrequentRecommender...")
        self.location_popularity = train_df['location_idx'].value_counts()
        self.train_user_items_map = train_user_items_map
        return self

    def recommend(self, user_idx, n=10):
        user_train_items = self.train_user_items_map.get(user_idx, set())
        # Recommend most popular items not seen in training
        recommendations = self.location_popularity[~self.location_popularity.index.isin(user_train_items)].head(n).index.tolist()
        return recommendations

class ItemBasedCFRecommender:
    def __init__(self):
        self.item_similarity = None
        self.train_user_items_map = None
        self.item_encoder = None
        self.user_item_matrix = None

    def fit(self, train_df, n_users, n_items, train_user_items_map, location_encoder):
        print("Fitting ItemBasedCFRecommender...")
        start_time = time.time()
        self.train_user_items_map = train_user_items_map
        self.item_encoder = location_encoder # Store encoder for potential inverse transform if needed

        # Create user-item matrix (users x items)
        user_item_data = np.ones(len(train_df))
        self.user_item_matrix = csr_matrix((user_item_data, (train_df['user_idx'], train_df['location_idx'])), shape=(n_users, n_items))

        # Calculate item-item similarity (cosine similarity on item vectors (columns))
        # Need items x items matrix, so transpose user_item_matrix (items x users) before similarity calculation
        print(" Calculating item-item similarity...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T, dense_output=False) # Keep sparse
        print(f" Item similarity matrix shape: {self.item_similarity.shape}")
        print(f" ItemBasedCF fitting done in {time.time() - start_time:.2f} seconds.")
        return self

    def recommend(self, user_idx, n=10):
        user_train_items = self.train_user_items_map.get(user_idx, set())
        if not user_train_items:
            return [] # Cannot recommend if user has no training interactions

        # Get user's interaction vector (sparse row)
        user_vector = self.user_item_matrix[user_idx, :]

        # Calculate scores for all items based on similarity to items user interacted with
        # Scores = UserVector * ItemSimilarityMatrix
        scores = user_vector.dot(self.item_similarity).toarray().ravel() # Convert sparse result to dense array

        # Rank items by score (descending)
        ranked_item_indices = np.argsort(scores)[::-1]

        # Filter out items already seen in training and take top N
        recommendations = []
        for item_idx in ranked_item_indices:
            if item_idx not in user_train_items:
                recommendations.append(item_idx)
            if len(recommendations) == n:
                break
        return recommendations

class UserBasedCFRecommender:
    def __init__(self):
        self.user_similarity = None
        self.train_user_items_map = None
        self.user_encoder = None
        self.user_item_matrix = None

    def fit(self, train_df, n_users, n_items, train_user_items_map, user_encoder):
        print("Fitting UserBasedCFRecommender...")
        start_time = time.time()
        self.train_user_items_map = train_user_items_map
        self.user_encoder = user_encoder # Store encoder

        # Create user-item matrix (users x items)
        user_item_data = np.ones(len(train_df))
        self.user_item_matrix = csr_matrix((user_item_data, (train_df['user_idx'], train_df['location_idx'])), shape=(n_users, n_items))

        # Calculate user-user similarity (cosine similarity on user vectors (rows))
        print(" Calculating user-user similarity...")
        self.user_similarity = cosine_similarity(self.user_item_matrix, dense_output=False)
        print(f" User similarity matrix shape: {self.user_similarity.shape}")
        print(f" UserBasedCF fitting done in {time.time() - start_time:.2f} seconds.")
        return self

    def recommend(self, user_idx, n=10, neighborhood_size=50):
        user_train_items = self.train_user_items_map.get(user_idx, set())

        # Get similarities for the target user
        user_sim_vector = self.user_similarity[user_idx, :].toarray().ravel()
        # Find top N similar users (excluding the user itself)
        user_sim_vector[user_idx] = -1 # Exclude self
        similar_user_indices = np.argsort(user_sim_vector)[::-1][:neighborhood_size]

        # Aggregate items from similar users, weighted by similarity
        item_scores = defaultdict(float)
        for sim_user_idx in similar_user_indices:
            similarity_score = user_sim_vector[sim_user_idx]
            if similarity_score <= 0: # Ignore non-positive similarity
                continue
            neighbor_items = self.train_user_items_map.get(sim_user_idx, set())
            for item_idx in neighbor_items:
                if item_idx not in user_train_items: # Only consider items target user hasn't seen
                    item_scores[item_idx] += similarity_score

        # Rank items by aggregated score
        ranked_items = sorted(item_scores.items(), key=lambda item: item[1], reverse=True)

        # Get top N recommendations
        recommendations = [item_idx for item_idx, score in ranked_items[:n]]
        return recommendations

class GeoUserBasedCFRecommender:
    def __init__(self):
        self.user_similarity = None
        self.train_user_items_map = None
        self.user_encoder = None
        self.user_item_matrix = None
        self.all_item_indices = None
        self.alpha = None
        self.n_users = None
        self.user_centroids = None

    def fit(self, train_df, n_users, n_items, train_user_items_map, user_encoder):
        print("Fitting GeoUserBasedCFRecommender...")
        start_time = time.time()
        self.train_user_items_map = train_user_items_map
        self.user_encoder = user_encoder
        self.all_item_indices = train_df['location_idx'].unique()
        self.alpha = GEO_WEIGHT_ALPHA # Weight for geographical influence
        self.n_users = n_users

        # Create user-item matrix
        rows = train_df['user_idx'].values
        cols = train_df['location_idx'].values
        data = np.ones(len(rows))
        self.user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Calculate user-user cosine similarity
        print(" Calculating user-user cosine similarity...")
        self.user_cosine_similarity = cosine_similarity(self.user_item_matrix, dense_output=False)
        print(f" User cosine similarity matrix shape: {self.user_cosine_similarity.shape}")

        # Calculate user centroids (average lat/lon)
        print(" Calculating user centroids...")
        # Ensure latitude and longitude are numeric, coercing errors
        train_df_geo = train_df.copy()
        train_df_geo['latitude'] = pd.to_numeric(train_df_geo['latitude'], errors='coerce')
        train_df_geo['longitude'] = pd.to_numeric(train_df_geo['longitude'], errors='coerce')
        train_df_geo = train_df_geo.dropna(subset=['latitude', 'longitude'])
        
        user_centroids = train_df_geo.groupby('user_idx')[['latitude', 'longitude']].mean().to_dict('index')
        # Store centroids as a dictionary: {user_idx: {'lat': lat, 'lon': lon}}
        self.user_centroids = {idx: {'lat': coords['latitude'], 'lon': coords['longitude']} 
                               for idx, coords in user_centroids.items()}
        print(f" Calculated centroids for {len(self.user_centroids)} users.")
        
        print(f"GeoUserBasedCF fitting done in {time.time() - start_time:.2f} seconds.")

    def recommend(self, user_idx, n=10, k=50):
        user_seen_items = set(self.train_user_items_map.get(user_idx, []))
        target_centroid = self.user_centroids.get(user_idx)

        if target_centroid is None:
            # Fallback: Use standard UserBasedCF logic if centroid is missing for the target user
            return self._recommend_cosine_only(user_idx, n, k)

        # Get cosine similarities for the target user
        user_cos_sim_vector = self.user_cosine_similarity[user_idx]
        user_cos_sim_vector = np.asarray(user_cos_sim_vector.todense()).flatten()
        
        # Calculate hybrid similarity for all other users
        hybrid_similarities = {}
        for neighbor_idx in range(self.n_users): # Iterate through all possible users
            if neighbor_idx == user_idx:
                continue
            
            neighbor_centroid = self.user_centroids.get(neighbor_idx)
            cosine_sim = user_cos_sim_vector[neighbor_idx]

            if neighbor_centroid:
                distance = haversine(target_centroid['lon'], target_centroid['lat'], 
                                     neighbor_centroid['lon'], neighbor_centroid['lat'])
                # Normalize distance slightly: use 1 / (1 + distance) to avoid large values for close users and division by zero
                geo_similarity = 1.0 / (1.0 + distance) 
                # Weighted combination
                hybrid_sim = (1 - self.alpha) * cosine_sim + self.alpha * geo_similarity
                hybrid_similarities[neighbor_idx] = hybrid_sim
            else:
                 # If neighbor has no centroid, just use weighted cosine similarity
                 hybrid_similarities[neighbor_idx] = (1 - self.alpha) * cosine_sim 
        
        # Find top K neighbors based on hybrid similarity
        sorted_neighbors = sorted(hybrid_similarities.items(), key=lambda item: item[1], reverse=True)
        similar_neighbors = sorted_neighbors[:k]
        
        if not similar_neighbors:
            return [] # No neighbors found

        # Aggregate items from hybrid neighbors, weighted by hybrid similarity
        item_scores = {} 
        total_similarity_sum = 0 # For potential normalization if needed
        for neighbor_idx, similarity in similar_neighbors:
            neighbor_items = self.train_user_items_map.get(neighbor_idx, [])
            for item_idx in neighbor_items:
                if item_idx not in user_seen_items: # Only recommend unseen items
                    item_scores[item_idx] = item_scores.get(item_idx, 0) + similarity
                    # total_similarity_sum += similarity # Optional normalization

        # Sort items by aggregated score
        # Optional: Normalize scores: item_scores = {item: score / total_similarity_sum for item, score in item_scores.items()} 
        sorted_items = sorted(item_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Get top N recommendations
        recommendations = [item_idx for item_idx, score in sorted_items[:n]]
        return recommendations
    
    def _recommend_cosine_only(self, user_idx, n, k):
        # Fallback logic using only cosine similarity (similar to UserBasedCFRecommender.recommend)
        user_seen_items = set(self.train_user_items_map.get(user_idx, []))
        user_sim_vector = self.user_cosine_similarity[user_idx]
        user_sim_vector = np.asarray(user_sim_vector.todense()).flatten()
        neighbor_indices = np.argsort(user_sim_vector)[::-1]
        similar_neighbors = []
        for neighbor_idx in neighbor_indices:
            if neighbor_idx != user_idx:
                similar_neighbors.append((neighbor_idx, user_sim_vector[neighbor_idx]))
            if len(similar_neighbors) == k:
                break
        if not similar_neighbors: return []
        item_scores = {} 
        for neighbor_idx, similarity in similar_neighbors:
            neighbor_items = self.train_user_items_map.get(neighbor_idx, [])
            for item_idx in neighbor_items:
                if item_idx not in user_seen_items:
                    item_scores[item_idx] = item_scores.get(item_idx, 0) + similarity
        sorted_items = sorted(item_scores.items(), key=lambda item: item[1], reverse=True)
        return [item_idx for item_idx, score in sorted_items[:n]]

# --- Evaluation ---
def evaluate_model(model, test_df, test_user_item_set, train_user_items_map, n=10):
    """Calculates Precision@N and Recall@N for a given model."""
    print(f"Evaluating {model.__class__.__name__}...")
    start_time = time.time()
    hits = 0
    total_precision = 0.0
    total_recall = 0.0
    test_user_count = len(test_df)

    for user_idx in test_df['user_idx'].unique():
        true_item_idx = test_df[test_df['user_idx'] == user_idx]['location_idx'].iloc[0]
        if (user_idx, true_item_idx) not in test_user_item_set: # Should not happen with current split, but good check
             continue

        recommendations = model.recommend(user_idx, n=n)

        if recommendations: # Check if recommendations list is not empty
            num_hit = int(true_item_idx in recommendations)
            hits += num_hit
            total_precision += num_hit / len(recommendations)
            total_recall += num_hit / 1 # Recall is defined wrt 1 relevant item in test set

    avg_precision = total_precision / test_user_count if test_user_count > 0 else 0.0
    avg_recall = total_recall / test_user_count if test_user_count > 0 else 0.0
    print(f" Evaluation done in {time.time() - start_time:.2f} seconds.")
    return avg_precision, avg_recall

def plot_comparison(results, output_dir):
    """Plots a bar chart comparing model performance."""
    print("Plotting model comparison...")
    model_names = list(results.keys())
    precision_scores = [res['precision'] for res in results.values()]
    recall_scores = [res['recall'] for res in results.values()]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, precision_scores, width, label=f'Precision@{RECOMMENDATION_K}')
    rects2 = ax.bar(x + width/2, recall_scores, width, label=f'Recall@{RECOMMENDATION_K}')

    ax.set_ylabel('Score')
    ax.set_title(f'Model Comparison (Precision@{RECOMMENDATION_K} & Recall@{RECOMMENDATION_K})')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    try:
        plt.savefig(plot_path)
        print(f" Model comparison plot saved to: {plot_path}")
    except Exception as e:
        print(f" Error saving comparison plot: {e}")
    plt.close()

# --- Main Workflow ---
if __name__ == '__main__':
    # 1. Load and Preprocess Data
    df, user_encoder, location_encoder = load_and_preprocess_data(
        DATA_FILE_PATH,
        nrows=SAMPLE_SIZE,
        min_user_checkins=MIN_USER_CHECKINS,
        min_loc_checkins=MIN_LOCATION_CHECKINS
    )

    if df is not None:
        n_users = df['user_idx'].nunique()
        n_items = df['location_idx'].nunique()
        print(f"Data ready: {n_users} users, {n_items} items after filtering.")

        # 2. EDA Plots (on filtered data)
        plot_distribution(df, 'user_id', 'Distribution of Check-ins per User (Filtered, Log Scale)', 'user_checkin_distribution_filtered.png', OUTPUT_DIR, log_scale=True)
        plot_distribution(df, 'location_id', 'Distribution of Check-ins per Location (Filtered, Log Scale)', 'location_checkin_distribution_filtered.png', OUTPUT_DIR, log_scale=True)
        plot_checkin_heatmap(df, OUTPUT_DIR) 
        
        # 3. Train-Test Split
        train_df, test_df, test_user_item_set, train_user_items_map = temporal_train_test_split(df)
        
        # Plot footprint for the first user *in the training dataset* after split
        if not train_df.empty:
            # Get the first original user ID from the training set
            first_user_idx = train_df['user_idx'].iloc[0] 
            first_user_id = user_encoder.inverse_transform([first_user_idx])[0]
            plot_user_footprint(train_df, first_user_id, user_encoder, OUTPUT_DIR) # Pass train_df
        else:
            print("Training data is empty, cannot plot user footprint.")

        # 4. Initialize Models
        models = {
            'MostFrequent': MostFrequentRecommender(),
            'ItemBasedCF': ItemBasedCFRecommender(),
            'UserBasedCF': UserBasedCFRecommender(),
            'GeoUserBasedCF': GeoUserBasedCFRecommender()
        }

        # 5. Fit Models
        model_results = {}
        models['MostFrequent'].fit(train_df, train_user_items_map)
        models['ItemBasedCF'].fit(train_df, n_users, n_items, train_user_items_map, location_encoder)
        models['UserBasedCF'].fit(train_df, n_users, n_items, train_user_items_map, user_encoder)
        models['GeoUserBasedCF'].fit(train_df, n_users, n_items, train_user_items_map, user_encoder)

        # 6. Evaluate Models
        for name, model in models.items():
            precision, recall = evaluate_model(model, test_df, test_user_item_set, train_user_items_map, n=RECOMMENDATION_K)
            model_results[name] = {'precision': precision, 'recall': recall}
            print(f" {name}: Precision@{RECOMMENDATION_K} = {precision:.4f}, Recall@{RECOMMENDATION_K} = {recall:.4f}")

        # 7. Plot Comparison
        plot_comparison(model_results, OUTPUT_DIR)

        print("\n--- Final Results ---")
        for name, metrics in model_results.items():
            print(f" {name}: P@{RECOMMENDATION_K}={metrics['precision']:.4f}, R@{RECOMMENDATION_K}={metrics['recall']:.4f}")

    else:
        print("Pipeline aborted due to data loading/preprocessing error.")
