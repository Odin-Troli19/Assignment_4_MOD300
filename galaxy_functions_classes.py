"""
As asked in the assignment document, we put all the functions and classes that would be used in a .py file
some of the functions were giving an error when imported so we had to rewrite them in the Notebook again

This module has in it all the functions for:
- Generating Milky Way visualizations using mw_plot
- Converting images to RGB arrays
- Encoding pixel data into categories
- Clustering using K-Means and K-NN
- Visualizing cluster results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, List, Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Task 1 & 2: Milky Way Visualization Functions
# =============================================================================

def generate_milky_way_facedown(
    radius: float = 20,
    title: str = "Bird's Eyes View",
    figsize: Tuple[int, int] = (10, 8),
    annotation: bool = True
) -> Figure:
    """
    Generate a face-on view of the Milky Way galaxy.
    
    Parameters
    ----------
    radius : float
        Radius of the view in kpc (kiloparsecs)
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    annotation : bool
        Whether to include annotations
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    from astropy import units as u
    from mw_plot import MWFaceOn
    
    mw = MWFaceOn(
        radius=radius * u.kpc,
        unit=u.kpc,
        coord="galactocentric",
        annotation=annotation,
        figsize=figsize,
    )
    mw.title = title
    # Mark the Sun's position (approximately 8 kpc from center)
    mw.scatter(8 * u.kpc, 0 * u.kpc, c="r", s=20, label="Sun")
    
    return mw


def generate_milky_way_skymap(
    center: str = "M31",
    radius: Tuple[float, float] = (8800, 8800),
    background: str = "Mellinger color optical survey",
    figsize: Tuple[int, int] = (8, 8)
) -> Tuple[Figure, plt.Axes]:
    """
    Generate a sky map view of a region of the Milky Way.
    
    Parameters
    ----------
    center : str
        Center of the observation (e.g., "M31", "galactic_center")
    radius : tuple
        Radius of observation in arcseconds (width, height)
    background : str
        Background survey to use
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    tuple
        (figure, axes) - The matplotlib figure and axes objects
    """
    from astropy import units as u
    from mw_plot import MWSkyMap
    
    mw = MWSkyMap(
        center=center,
        radius=radius * u.arcsec,
        background=background,
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    mw.transform(ax)
    
    return fig, ax, mw


def generate_multiple_views(
    centers: List[str] = ["M31", "galactic_center", "M42"],
    radii: List[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (6, 6)
) -> List[Tuple[Figure, plt.Axes]]:
    """
    Generate multiple sky map views with different centers and radii.
    
    Parameters
    ----------
    centers : list of str
        List of center points for observations
    radii : list of tuples
        List of radii in arcseconds for each center
    figsize : tuple
        Figure size for each plot
        
    Returns
    -------
    list of tuples
        List of (figure, axes, mw_object) for each view
    """
    from astropy import units as u
    from mw_plot import MWSkyMap
    
    if radii is None:
        radii = [(8800, 8800)] * len(centers)
    
    assert len(centers) == len(radii), "Centers and radii must have same length"
    
    results = []
    for center, radius in zip(centers, radii):
        try:
            mw = MWSkyMap(
                center=center,
                radius=radius * u.arcsec,
                background="Mellinger color optical survey",
            )
            fig, ax = plt.subplots(figsize=figsize)
            mw.transform(ax)
            ax.set_title(f"Center: {center}, Radius: {radius}")
            results.append((fig, ax, mw))
        except Exception as e:
            print(f"Could not generate view for {center}: {e}")
            
    return results


# =============================================================================
# Task 3: Image to RGB Array Conversion
# =============================================================================

def figure_to_rgb_array(fig: Figure) -> np.ndarray:
    """
    Convert a matplotlib figure to a 3D RGB numpy array.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert
        
    Returns
    -------
    np.ndarray
        3D array of shape (height, width, 3) with RGB values (0-255)
    """
    assert isinstance(fig, Figure), "Input must be a matplotlib Figure"
    
    # Remove padding around the axes
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    
    # Get the RGBA buffer
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    
    # Convert to numpy array and extract RGB (drop alpha channel)
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    rgb_arr = rgba_arr[:, :, :3].copy()
    
    assert rgb_arr.ndim == 3, "Output must be 3-dimensional"
    assert rgb_arr.shape[2] == 3, "Third dimension must be 3 (RGB)"
    
    return rgb_arr


def load_image_as_rgb(filepath: str) -> np.ndarray:
    """
    Load an image file and convert to RGB array.
    
    Parameters
    ----------
    filepath : str
        Path to the image file
        
    Returns
    -------
    np.ndarray
        3D array of shape (height, width, 3) with RGB values
    """
    from PIL import Image
    
    img = Image.open(filepath).convert('RGB')
    return np.array(img)


# =============================================================================
# Task 4: Encoding Functions (Converting RGB to Categories)
# =============================================================================

def encode_grey(rgb_array: np.ndarray) -> np.ndarray:
    """
    Encode RGB array to grayscale using standard weights.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D array of shape (height, width, 3) with RGB values
        
    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with grayscale values (0-255)
    """
    weights = np.array([0.299, 0.587, 0.114])
    grey = np.sum(rgb_array * weights, axis=2)
    return grey


def encode_brightness_category(
    rgb_array: np.ndarray,
    thresholds: List[int] = [50, 100, 150, 200]
) -> np.ndarray:
    """
    Encode pixels into brightness categories.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    thresholds : list of int
        Brightness thresholds for categories
        
    Returns
    -------
    np.ndarray
        2D array with category labels (0 to len(thresholds))
    """
    grey = encode_grey(rgb_array)
    categories = np.zeros_like(grey, dtype=int)
    
    for i, threshold in enumerate(thresholds):
        categories[grey > threshold] = i + 1
        
    return categories


def encode_color_category(rgb_array: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Encode pixels into color categories based on dominant channel.
    
    Categories:
    - 0: Dark (all channels below threshold)
    - 1: Red dominant
    - 2: Green dominant  
    - 3: Blue dominant
    - 4: Yellow (R+G dominant)
    - 5: Cyan (G+B dominant)
    - 6: Magenta (R+B dominant)
    - 7: White/Bright (all channels high)
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
        
    Returns
    -------
    tuple
        (category_array, category_labels_dict)
    """
    assert rgb_array.ndim == 3 and rgb_array.shape[2] == 3
    
    r, g, b = rgb_array[:,:,0], rgb_array[:,:,1], rgb_array[:,:,2]
    
    categories = np.zeros(rgb_array.shape[:2], dtype=int)
    
    # Define thresholds
    dark_threshold = 50
    bright_threshold = 200
    
    # Total brightness
    brightness = encode_grey(rgb_array)
    
    # Dark pixels
    categories[brightness < dark_threshold] = 0
    
    # White/bright pixels
    is_bright = (r > bright_threshold) & (g > bright_threshold) & (b > bright_threshold)
    categories[is_bright] = 7
    
    # For remaining pixels, determine dominant color
    mask_not_extreme = ~is_bright & (brightness >= dark_threshold)
    
    # Red dominant
    red_dominant = mask_not_extreme & (r > g) & (r > b)
    categories[red_dominant] = 1
    
    # Green dominant
    green_dominant = mask_not_extreme & (g > r) & (g > b)
    categories[green_dominant] = 2
    
    # Blue dominant
    blue_dominant = mask_not_extreme & (b > r) & (b > g)
    categories[blue_dominant] = 3
    
    # Yellow (R+G both high, B low)
    yellow = mask_not_extreme & (r > 100) & (g > 100) & (b < 100)
    categories[yellow] = 4
    
    # Cyan (G+B both high, R low)
    cyan = mask_not_extreme & (g > 100) & (b > 100) & (r < 100)
    categories[cyan] = 5
    
    # Magenta (R+B both high, G low)
    magenta = mask_not_extreme & (r > 100) & (b > 100) & (g < 100)
    categories[magenta] = 6
    
    labels = {
        0: "Dark",
        1: "Red",
        2: "Green",
        3: "Blue",
        4: "Yellow",
        5: "Cyan",
        6: "Magenta",
        7: "White/Bright"
    }
    
    return categories, labels


def encode_custom(
    rgb_array: np.ndarray,
    encoding_type: str = "grey"
) -> Tuple[np.ndarray, dict]:
    """
    Apply custom encoding based on type selection.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    encoding_type : str
        Type of encoding: "grey", "brightness", "color", or "intensity"
        
    Returns
    -------
    tuple
        (encoded_array, labels_dict)
    """
    if encoding_type == "grey":
        encoded = encode_grey(rgb_array)
        labels = {"description": "Grayscale intensity 0-255"}
        
    elif encoding_type == "brightness":
        encoded = encode_brightness_category(rgb_array)
        labels = {
            0: "Very Dark (0-50)",
            1: "Dark (50-100)",
            2: "Medium (100-150)",
            3: "Bright (150-200)",
            4: "Very Bright (200-255)"
        }
        
    elif encoding_type == "color":
        encoded, labels = encode_color_category(rgb_array)
        
    elif encoding_type == "intensity":
        # Encode based on overall intensity with 3 levels
        grey = encode_grey(rgb_array)
        encoded = np.zeros_like(grey, dtype=int)
        encoded[grey > 85] = 1
        encoded[grey > 170] = 2
        labels = {0: "Low", 1: "Medium", 2: "High"}
        
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
        
    return encoded, labels


# =============================================================================
# Task 5: Clustering Functions (K-Means and K-NN)
# =============================================================================

def prepare_pixel_features(
    rgb_array: np.ndarray,
    include_position: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare pixel data as feature vectors for clustering.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    include_position : bool
        Whether to include (x, y) position as features
        
    Returns
    -------
    tuple
        (features, x_coords, y_coords) - Feature matrix and coordinate arrays
    """
    h, w = rgb_array.shape[:2]
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Flatten everything
    r = rgb_array[:,:,0].flatten()
    g = rgb_array[:,:,1].flatten()
    b = rgb_array[:,:,2].flatten()
    x = x_coords.flatten()
    y = y_coords.flatten()
    
    if include_position:
        # Normalize position to same scale as color values
        x_norm = (x / w) * 255
        y_norm = (y / h) * 255
        features = np.column_stack([r, g, b, x_norm, y_norm])
    else:
        features = np.column_stack([r, g, b])
        
    return features, x, y


def cluster_kmeans(
    rgb_array: np.ndarray,
    n_clusters: int = 5,
    include_position: bool = False,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-Means clustering on pixel data.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    n_clusters : int
        Number of clusters
    include_position : bool
        Whether to include pixel position in features
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (cluster_labels_2d, kmeans_model)
    """
    assert n_clusters > 0, "Number of clusters must be positive"
    
    features, x, y = prepare_pixel_features(rgb_array, include_position)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    # Reshape labels to image dimensions
    h, w = rgb_array.shape[:2]
    labels_2d = labels.reshape(h, w)
    
    return labels_2d, kmeans


def cluster_knn(
    rgb_array: np.ndarray,
    training_labels: np.ndarray,
    training_mask: np.ndarray,
    n_neighbors: int = 5
) -> Tuple[np.ndarray, KNeighborsClassifier]:
    """
    Perform K-NN classification on pixel data using labeled training points.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    training_labels : np.ndarray
        2D array with labels for training pixels
    training_mask : np.ndarray
        2D boolean array indicating which pixels are training data
    n_neighbors : int
        Number of neighbors for K-NN
        
    Returns
    -------
    tuple
        (predicted_labels_2d, knn_model)
    """
    features, x, y = prepare_pixel_features(rgb_array, include_position=True)
    
    # Get training data
    train_mask_flat = training_mask.flatten()
    X_train = features[train_mask_flat]
    y_train = training_labels.flatten()[train_mask_flat]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(features)
    
    # Train K-NN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    
    # Predict all pixels
    labels = knn.predict(X_all_scaled)
    
    # Reshape to image dimensions
    h, w = rgb_array.shape[:2]
    labels_2d = labels.reshape(h, w)
    
    return labels_2d, knn


def cluster_from_encoding(
    rgb_array: np.ndarray,
    encoding_type: str = "brightness",
    n_clusters: int = 5
) -> np.ndarray:
    """
    Cluster pixels based on their encoded values.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    encoding_type : str
        Type of encoding to use before clustering
    n_clusters : int
        Number of clusters for K-Means
        
    Returns
    -------
    np.ndarray
        2D array of cluster labels
    """
    # Get encoded values
    encoded, _ = encode_custom(rgb_array, encoding_type)
    
    # Flatten and reshape for clustering
    h, w = encoded.shape
    features = encoded.flatten().reshape(-1, 1)
    
    # Perform K-Means on encoded values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return labels.reshape(h, w)


# =============================================================================
# Task 6 & 7: Visualization Functions
# =============================================================================

def visualize_clusters(
    cluster_labels: np.ndarray,
    title: str = "Cluster Visualization",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis"
) -> Figure:
    """
    Visualize cluster labels as a colored image.
    
    Parameters
    ----------
    cluster_labels : np.ndarray
        2D array of cluster labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cluster_labels, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    plt.colorbar(im, ax=ax, label="Cluster ID")
    
    return fig


def overlay_clusters_on_image(
    original_fig: Figure,
    cluster_labels: np.ndarray,
    alpha: float = 0.5,
    title: str = "Clusters Overlaid on Galaxy",
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Overlay cluster visualization on the original galaxy image.
    
    Parameters
    ----------
    original_fig : matplotlib.figure.Figure
        Original galaxy figure
    cluster_labels : np.ndarray
        2D array of cluster labels
    alpha : float
        Transparency of cluster overlay (0-1)
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Get RGB array from original figure
    original_rgb = figure_to_rgb_array(original_fig)
    
    # Create new figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show original image
    ax.imshow(original_rgb)
    
    # Overlay clusters with transparency
    ax.imshow(cluster_labels, cmap='tab10', alpha=alpha)
    
    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    
    return fig


def overlay_clusters_scatter(
    rgb_array: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Cluster Points Overlaid",
    figsize: Tuple[int, int] = (10, 8),
    sample_fraction: float = 0.1
) -> Figure:
    """
    Overlay cluster points as scatter on original image.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        Original RGB array
    cluster_labels : np.ndarray
        2D array of cluster labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    sample_fraction : float
        Fraction of points to plot (for performance)
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show original image
    ax.imshow(rgb_array)
    
    # Get coordinates for each cluster
    h, w = cluster_labels.shape
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        y_coords, x_coords = np.where(cluster_labels == cluster_id)
        
        # Sample points for faster plotting
        n_points = len(x_coords)
        n_sample = max(1, int(n_points * sample_fraction))
        indices = np.random.choice(n_points, n_sample, replace=False)
        
        ax.scatter(x_coords[indices], y_coords[indices], 
                   c=[colors[cluster_id]], s=1, alpha=0.5,
                   label=f"Cluster {cluster_id}")
    
    ax.set_title(title)
    ax.legend(loc='upper right', markerscale=5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    return fig


def compare_encodings(
    rgb_array: np.ndarray,
    encoding_types: List[str] = ["grey", "brightness", "color"],
    figsize: Tuple[int, int] = (15, 5)
) -> Figure:
    """
    Compare different encoding methods side by side.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    encoding_types : list of str
        List of encoding types to compare
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(encoding_types)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, enc_type in zip(axes, encoding_types):
        encoded, labels = encode_custom(rgb_array, enc_type)
        im = ax.imshow(encoded, cmap='viridis')
        ax.set_title(f"Encoding: {enc_type}")
        plt.colorbar(im, ax=ax)
        
    fig.tight_layout()
    return fig


def compare_cluster_counts(
    rgb_array: np.ndarray,
    cluster_range: List[int] = [3, 5, 7, 10],
    figsize: Tuple[int, int] = (15, 10)
) -> Figure:
    """
    Compare clustering results with different numbers of clusters.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    cluster_range : list of int
        Different cluster counts to compare
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(cluster_range)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, n_clusters in enumerate(cluster_range):
        labels, _ = cluster_kmeans(rgb_array, n_clusters=n_clusters)
        axes[i].imshow(labels, cmap='tab10')
        axes[i].set_title(f"K-Means: {n_clusters} clusters")
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.tight_layout()
    return fig


def extract_bright_pixels(
    rgb_array: np.ndarray,
    threshold: int = 230
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract coordinates of bright pixels (like in the original notebook).
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    threshold : int
        Brightness threshold (0-255)
        
    Returns
    -------
    tuple
        (x_coords, y_coords) arrays
    """
    grey = encode_grey(rgb_array)
    y_coords, x_coords = np.where(grey > threshold)
    return x_coords, y_coords


def plot_bright_pixels(
    rgb_array: np.ndarray,
    threshold: int = 230,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Plot bright pixel positions as a scatter plot.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        3D RGB array
    threshold : int
        Brightness threshold
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    x, y = extract_bright_pixels(rgb_array, threshold)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y, x, s=0.1, c='white')
    ax.set_facecolor('black')
    ax.set_title(f"Bright Pixels (threshold > {threshold})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def save_figure(fig: Figure, filepath: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to file."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")


def get_cluster_statistics(cluster_labels: np.ndarray) -> dict:
    """
    Calculate statistics about cluster distribution.
    
    Parameters
    ----------
    cluster_labels : np.ndarray
        2D array of cluster labels
        
    Returns
    -------
    dict
        Dictionary with cluster statistics
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    total = cluster_labels.size
    
    stats = {
        "n_clusters": len(unique),
        "total_pixels": total,
        "cluster_counts": dict(zip(unique.tolist(), counts.tolist())),
        "cluster_percentages": {
            int(u): round(100 * c / total, 2) 
            for u, c in zip(unique, counts)
        }
    }
    
    return stats


def print_cluster_summary(stats: dict) -> None:
    """Print a summary of cluster statistics."""
    print(f"\n{'='*50}")
    print("CLUSTER ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Number of clusters: {stats['n_clusters']}")
    print(f"Total pixels: {stats['total_pixels']:,}")
    print(f"\nCluster distribution:")
    for cluster_id, percentage in stats['cluster_percentages'].items():
        count = stats['cluster_counts'][cluster_id]
        print(f"  Cluster {cluster_id}: {count:,} pixels ({percentage:.2f}%)")
    print(f"{'='*50}\n")
