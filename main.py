"""
Computer Vision in AI - Demo Suite
===================================
Hands-on demonstrations of core CV techniques covered in the notebook.

Usage:
    python main.py                      # Run all demos
    python main.py --demo edges         # Edge detection
    python main.py --demo features      # Feature extraction & matching
    python main.py --demo segmentation  # Color-based segmentation
    python main.py --demo augment       # Data augmentation pipeline
    python main.py --demo contrastive   # Contrastive similarity
    python main.py --demo gan           # GAN architecture overview
    python main.py --demo xai           # Gradient saliency map
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; change to "TkAgg" for live display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _synthetic_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Generate a simple synthetic BGR image with geometric shapes."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Background gradient
    for i in range(h):
        img[i, :] = [int(i * 0.5), 30, int((h - i) * 0.5)]
    # White rectangle
    img[60:100, 60:196] = [220, 220, 220]
    # Red circle region (approximate)
    cy, cx, r = 160, 128, 40
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = [0, 0, 200]
    return img


# ---------------------------------------------------------------------------
# Demo 1: Edge Detection (Canny)
# ---------------------------------------------------------------------------

def demo_edges() -> None:
    """Edge detection pipeline using gradient-based approach (no OpenCV needed)."""
    print("\n[1/7] Edge Detection Demo")
    try:
        import cv2
        img = _synthetic_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[1].imshow(gray, cmap="gray")
        axes[1].set_title("Grayscale")
        axes[2].imshow(edges, cmap="gray")
        axes[2].set_title("Canny Edges")
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Edge Detection", fontsize=14, fontweight="bold")
        _save(fig, "demo_edges")
    except ImportError:
        print("  OpenCV not installed. Install with: pip install opencv-python")


# ---------------------------------------------------------------------------
# Demo 2: Feature Extraction (ORB keypoints)
# ---------------------------------------------------------------------------

def demo_features() -> None:
    """ORB keypoint detection — lightweight, no GPU needed."""
    print("\n[2/7] Feature Extraction Demo")
    try:
        import cv2
        img = _synthetic_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=50)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        vis = cv2.drawKeypoints(img, keypoints, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"ORB Keypoints ({len(keypoints)} detected)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Feature Extraction (ORB)", fontsize=14, fontweight="bold")
        _save(fig, "demo_features")
        print(f"  Detected {len(keypoints)} keypoints, descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    except ImportError:
        print("  OpenCV not installed. Install with: pip install opencv-python")


# ---------------------------------------------------------------------------
# Demo 3: Color-Based Segmentation (HSV thresholding)
# ---------------------------------------------------------------------------

def demo_segmentation() -> None:
    """Segment an image region by color using HSV thresholding."""
    print("\n[3/7] Color Segmentation Demo")
    try:
        import cv2
        img = _synthetic_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red hue range in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Also capture the upper red range
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask, mask2)

        result = cv2.bitwise_and(img, img, mask=mask)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Red Mask")
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Segmented (Red)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Color-Based Segmentation", fontsize=14, fontweight="bold")
        _save(fig, "demo_segmentation")
    except ImportError:
        print("  OpenCV not installed. Install with: pip install opencv-python")


# ---------------------------------------------------------------------------
# Demo 4: Data Augmentation Pipeline
# ---------------------------------------------------------------------------

def demo_augment() -> None:
    """Show common augmentations: flip, rotate, brightness, noise."""
    print("\n[4/7] Data Augmentation Demo")
    try:
        import cv2
        img = _synthetic_image()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def flip_h(x):
            return np.fliplr(x)

        def rotate(x, angle=30):
            h, w = x.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(x, M, (w, h))

        def brightness(x, factor=1.5):
            return np.clip(x.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        def add_noise(x, sigma=20):
            noise = np.random.normal(0, sigma, x.shape).astype(np.int16)
            return np.clip(x.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        augmented = [
            ("Original", rgb),
            ("Horizontal Flip", flip_h(rgb)),
            ("Rotation 30°", rotate(rgb)),
            ("Brightness +50%", brightness(rgb)),
            ("Gaussian Noise", add_noise(rgb)),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for ax, (title, im) in zip(axes, augmented):
            ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle("Data Augmentation Pipeline", fontsize=14, fontweight="bold")
        _save(fig, "demo_augment")
    except ImportError:
        print("  OpenCV not installed. Install with: pip install opencv-python")


# ---------------------------------------------------------------------------
# Demo 5: Contrastive Learning — Cosine Similarity
# ---------------------------------------------------------------------------

def demo_contrastive() -> None:
    """
    Simulate contrastive learning: compare feature vectors of augmented
    (positive) vs. random (negative) image pairs using cosine similarity.
    """
    print("\n[5/7] Contrastive Learning Demo")
    np.random.seed(42)

    def extract_features(img: np.ndarray) -> np.ndarray:
        """Simulate a CNN encoder: flatten + normalize."""
        flat = img.astype(np.float32).flatten()
        norm = np.linalg.norm(flat)
        return flat / (norm + 1e-8)

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    base = _synthetic_image(64, 64)
    # Positive pair: slight noise (same scene augmentation)
    positive = np.clip(base.astype(np.int16) + np.random.normal(0, 5, base.shape), 0, 255).astype(np.uint8)
    # Negative pair: completely random image
    negative = np.random.randint(0, 256, base.shape, dtype=np.uint8)

    f_base = extract_features(base)
    f_pos = extract_features(positive)
    f_neg = extract_features(negative)

    sim_pos = cosine_similarity(f_base, f_pos)
    sim_neg = cosine_similarity(f_base, f_neg)

    print(f"  Positive pair cosine similarity : {sim_pos:.4f}")
    print(f"  Negative pair cosine similarity : {sim_neg:.4f}")
    print(f"  Contrastive margin              : {sim_pos - sim_neg:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, im) in zip(axes, [("Anchor", base), ("Positive", positive), ("Negative", negative)]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    fig.text(0.5, 0.01,
             f"Anchor↔Positive: {sim_pos:.3f}   |   Anchor↔Negative: {sim_neg:.3f}",
             ha="center", fontsize=10, color="darkblue")
    fig.suptitle("Contrastive Learning: Cosine Similarity", fontsize=14, fontweight="bold")
    _save(fig, "demo_contrastive")


# ---------------------------------------------------------------------------
# Demo 6: GAN Architecture Overview (diagram, no training)
# ---------------------------------------------------------------------------

def demo_gan() -> None:
    """Visualize the GAN generator/discriminator training loop as a diagram."""
    print("\n[6/7] GAN Architecture Overview")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = {
        "Noise z\n(latent space)": (0.3, 2.0, 1.4, 1.0),
        "Generator\nG(z)": (2.2, 1.8, 1.6, 1.4),
        "Fake Images": (4.5, 2.0, 1.4, 1.0),
        "Real Images": (4.5, 3.5, 1.4, 1.0),
        "Discriminator\nD(x)": (6.8, 2.0, 1.6, 1.8),
        "Real / Fake\nPrediction": (9.0, 2.2, 0.9, 1.4),
    }
    colors = ["#AED6F1", "#A9DFBF", "#FAD7A0", "#F9E79F", "#D7BDE2", "#F1948A"]

    for (label, (x, y, w, h)), color in zip(boxes.items(), colors):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.1",
                                       linewidth=1.5, edgecolor="gray",
                                       facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8, fontweight="bold")

    # Arrows
    arrows = [
        ((1.7, 2.5), (2.2, 2.5)),       # noise → generator
        ((3.8, 2.5), (4.5, 2.5)),       # generator → fake
        ((5.9, 2.5), (6.8, 2.5)),       # fake → discriminator
        ((5.9, 4.0), (7.6, 3.8)),       # real → discriminator
        ((8.4, 2.9), (9.0, 2.9)),       # discriminator → prediction
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Gradient feedback label
    ax.annotate("", xy=(5.2, 1.8), xytext=(7.6, 1.8),
                arrowprops=dict(arrowstyle="<-", color="red", lw=1.5, linestyle="dashed"))
    ax.text(6.2, 1.5, "Gradient feedback\n(adversarial loss)", ha="center",
            fontsize=7, color="red")

    fig.suptitle("Generative Adversarial Network (GAN) — Architecture", fontsize=13, fontweight="bold")
    _save(fig, "demo_gan")


# ---------------------------------------------------------------------------
# Demo 7: Explainable AI — Gradient Saliency Map (numpy-only simulation)
# ---------------------------------------------------------------------------

def demo_xai() -> None:
    """
    Simulate a gradient-based saliency map (XAI).
    In practice this uses backprop gradients; here we approximate with
    a Sobel-like gradient magnitude on a synthetic image.
    """
    print("\n[7/7] Explainable AI (Saliency Map) Demo")
    try:
        import cv2

        img = _synthetic_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Sobel gradients as saliency proxy
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        saliency = np.sqrt(gx ** 2 + gy ** 2)
        saliency = (saliency / saliency.max() * 255).astype(np.uint8)

        # Overlay heatmap
        heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.55, heatmap, 0.45, 0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input Image")
        axes[1].imshow(saliency, cmap="hot")
        axes[1].set_title("Saliency Map\n(gradient magnitude)")
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Overlay (XAI)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Explainable AI — Gradient Saliency Map", fontsize=14, fontweight="bold")
        _save(fig, "demo_xai")
    except ImportError:
        print("  OpenCV not installed. Install with: pip install opencv-python")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEMOS = {
    "edges": demo_edges,
    "features": demo_features,
    "segmentation": demo_segmentation,
    "augment": demo_augment,
    "contrastive": demo_contrastive,
    "gan": demo_gan,
    "xai": demo_xai,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computer Vision in AI — Demo Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  --demo {k}" for k in DEMOS),
    )
    parser.add_argument(
        "--demo",
        choices=list(DEMOS.keys()),
        default=None,
        help="Run a specific demo (default: run all)",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Computer Vision in AI — Demo Suite")
    print("=" * 55)

    if args.demo:
        DEMOS[args.demo]()
    else:
        for fn in DEMOS.values():
            fn()

    print("\nDone. Output images saved in the current directory.")


if __name__ == "__main__":
    main()
