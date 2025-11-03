import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import string
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import animation
from sklearn.model_selection import train_test_split
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def create_semi_random_dataset(
    num_samples: int,
    num_features: int,
    num_clusters: int,
    percentage_to_change: float,
    random_seed: int
):
    np.random.seed(random_seed)
    letters = list(string.ascii_uppercase)

    # Create cluster centers (letters)
    true_centers = np.array([[np.random.choice(letters) for _ in range(num_features)] for _ in range(num_clusters)])

    # Assign samples to clusters
    true_labels = np.random.choice(num_clusters, size=num_samples)

    # Generate samples around their cluster centers (randomly swap letters in some positions)
    sample_list = []
    for i in range(num_samples):
        center = true_centers[true_labels[i]].copy()
        # Introduce a categorical variation by swapping letters in some positions
        swap_idx = np.random.choice(num_features, size=int(num_features * percentage_to_change), replace=False)
        for idx in swap_idx:
            center[idx] = np.random.choice(letters)
        sample_list.append(center)

    return np.array(sample_list), true_labels, true_centers

def jaccard_index(a: set[int], b: set[int]) -> float:
    # Compute Jaccard index between each cluster center and its assigned samples
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0

def visualize_jaccard_heatmap(jaccard_matrix: np.ndarray, filename: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(jaccard_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Average Jaccard Index')
    plt.title('Jaccard Index Heatmap (Clustered Samples vs. Centers)')
    plt.xlabel('Cluster Center Index')
    plt.ylabel('Assigned Cluster Index')
    plt.xticks(range(jaccard_matrix.shape[0]))
    plt.yticks(range(jaccard_matrix.shape[1]))
    plt.savefig(filename)
    plt.close()
    print(f"Jaccard heatmap saved to {filename}")

def compute_tsne_and_plot(x: np.ndarray, predicted_labels: np.ndarray, random_seed: int, num_components: int, filename: str):
    # Encode categorical data to numerical values
    encoder = OrdinalEncoder()
    x_encoded = encoder.fit_transform(x)

    # Apply t-SNE
    if num_components == 2:
        tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30, learning_rate=200)
        x_embedded = tsne.fit_transform(x_encoded)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=predicted_labels, cmap='tab20', s=10)
        plt.title("t-SNE visualization of semi-random clustered samples")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        cbar = plt.colorbar(scatter, ticks=np.arange(0, len(np.unique(predicted_labels)), 2))
        cbar.set_label("Cluster ID")
        plt.savefig(filename)
        print(f"t-SNE plot saved to {filename}")
    elif num_components == 3:
        tsne = TSNE(n_components=3, random_state=random_seed, perplexity=30, learning_rate=200)
        x_embedded = tsne.fit_transform(x_encoded)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            x_embedded[:, 0], x_embedded[:, 1], x_embedded[:, 2], c=predicted_labels, cmap='tab20', s=10
        )
        ax.set_title("t-SNE visualization of semi-random clustered samples (3D)")
        ax.grid(alpha=0.1)
        ax.set_facecolor((1, 1, 1, 0.1))  # RGBA: white with alpha=0.2
        fig.patch.set_alpha(0.1)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
        # Set grid and background transparency
        cbar = plt.colorbar(scatter, ticks=np.arange(0, len(np.unique(predicted_labels)), 2))
        cbar.set_label("Cluster ID")
        def rotate(angle):
            ax.view_init(elev=30, azim=angle)
        ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 36), interval=500)
        ani.save(filename.replace('.png', '.gif'), writer='pillow')
        print(f"Animated t-SNE 3D plot saved to {filename.replace('.png', '.gif')}")
        plt.close()
    else:
        raise ValueError("num_components must be either 2 or 3")

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Cluster semi-random categorical data and visualize with t-SNE.")
    parser.add_argument("--num_samples", type=int, default=8000, help="Number of samples to generate.")
    parser.add_argument("--num_features", type=int, default=5, help="Number of categorical features.")
    parser.add_argument("--num_clusters", type=int, default=20, help="Number of clusters.")
    parser.add_argument("--percentage_to_change", type=float, default=0.3, help="Percentage of features to randomly change in each sample.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_components", type=int, default=2, choices=[2, 3], help="Number of t-SNE components (2 or 3).")
    parser.add_argument("--run_tsne", action="store_true", help="Run t-SNE visualization.")
    parser.add_argument("--run_jaccard", action="store_true", help="Run Jaccard index heatmap.")
    parser.add_argument("--run_training", action="store_true", help="Run classifier training and evaluation.")
    parser.add_argument("--split_strategy", type=str, default="random", choices=["random"], help="Data split strategy for training/validation/testing.")
    args = parser.parse_args()

    # Step 1: Create a fake dataset
    x, true_labels, true_centers = create_semi_random_dataset(
        num_samples=args.num_samples,
        num_features=args.num_features,
        num_clusters=args.num_clusters,
        percentage_to_change=args.percentage_to_change,
        random_seed=args.random_seed
    )

    # Step 2: Cluster using KModes
    km = KModes(n_clusters=args.num_clusters, init='Huang', random_state=args.random_seed)
    predicted_labels = km.fit_predict(x)
    kmeans_centers = km.cluster_centroids_

    if args.run_jaccard:
        # Step 3: Compute Jaccard index matrix
        jaccard_matrix = np.zeros(shape=(args.num_clusters, args.num_clusters))
        for k in range(args.num_clusters):
            assigned_samples = x[predicted_labels == k]
            # center = np.round(kmeans_centers[k]).astype(int)
            for j in range(args.num_clusters):
                center = kmeans_centers[j]
                scores = [jaccard_index(set(center.tolist()), set(sample.tolist())) for sample in assigned_samples]
                avg_score = np.mean(scores) if scores else 0
                jaccard_matrix[k, j] = avg_score
        visualize_jaccard_heatmap(jaccard_matrix, "jaccard_heatmap.png")

    # Step 4: Compute t-SNE and plot
    if args.run_tsne:
        compute_tsne_and_plot(x, predicted_labels, args.random_seed, args.num_components, f"tsne_{args.num_components}.png")

    # Split into train, val, test (e.g., 70% train, 15% val, 15% test)
    if args.run_training:
        if args.split_strategy == 'random':
            x_train, x_temp, labels_train, labels_temp = train_test_split(
                x, predicted_labels, test_size=0.3, random_state=args.random_seed, stratify=predicted_labels
            )
            x_val, x_test, labels_val, labels_test = train_test_split(
                x_temp, labels_temp, test_size=0.5, random_state=args.random_seed, stratify=labels_temp
            )
        else:
            raise ValueError("Unsupported split strategy")

        print(f"Train samples: {len(x_train)}, Val samples: {len(x_val)}, Test samples: {len(x_test)}")
        # Encode categorical features for classifier
        encoder = OrdinalEncoder()
        x_train_enc = encoder.fit_transform(x_train)
        x_val_enc = encoder.transform(x_val)
        x_test_enc = encoder.transform(x_test)

        clf = RandomForestClassifier(random_state=args.random_seed)
        clf.fit(x_train_enc, labels_train)

        val_preds = clf.predict(x_val_enc)
        test_preds = clf.predict(x_test_enc)

        print(f"Validation accuracy: {accuracy_score(labels_val, val_preds):.4f}")
        print(f"Test accuracy: {accuracy_score(labels_test, test_preds):.4f}")
