"""
data_loader.py
--------------
Step 1: Load the dataset, explore it, and clean it.
Run this file first: python src/data_loader.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. LOAD DATA ──────────────────────────────────────────────
def load_data(filepath="data/student-mat.csv"):
    """Load the UCI Student Performance dataset."""
    if not os.path.exists(filepath):
        print("=" * 60)
        print("DATASET NOT FOUND — Follow these steps:")
        print("=" * 60)
        print("1. Go to: https://archive.ics.uci.edu/dataset/320/student+performance")
        print("2. Click 'Download Dataset'")
        print("3. Extract the zip file")
        print("4. Copy 'student-mat.csv' into the 'data/' folder")
        print("=" * 60)
        print()
        print("OR run this to generate sample data for now:")
        print("   python src/data_loader.py --sample")
        return None

    # The UCI file uses semicolon as separator
    df = pd.read_csv(filepath, sep=";")
    print(f"✓ Loaded dataset: {df.shape[0]} students, {df.shape[1]} features")
    return df


def generate_sample_data(n=395):
    """
    Generate realistic sample data that mirrors the UCI dataset structure.
    Use this while you wait to download the real dataset.
    """
    np.random.seed(42)

    data = {
        "school":      np.random.choice(["GP", "MS"], n),
        "sex":         np.random.choice(["M", "F"], n),
        "age":         np.random.randint(15, 23, n),
        "address":     np.random.choice(["U", "R"], n, p=[0.7, 0.3]),
        "famsize":     np.random.choice(["LE3", "GT3"], n),
        "Pstatus":     np.random.choice(["T", "A"], n, p=[0.8, 0.2]),
        "Medu":        np.random.randint(0, 5, n),   # Mother education 0-4
        "Fedu":        np.random.randint(0, 5, n),   # Father education 0-4
        "studytime":   np.random.randint(1, 5, n),   # Weekly study hours
        "failures":    np.random.choice([0,1,2,3], n, p=[0.67,0.18,0.1,0.05]),
        "schoolsup":   np.random.choice(["yes","no"], n, p=[0.3,0.7]),
        "famsup":      np.random.choice(["yes","no"], n, p=[0.6,0.4]),
        "paid":        np.random.choice(["yes","no"], n, p=[0.4,0.6]),
        "activities":  np.random.choice(["yes","no"], n),
        "higher":      np.random.choice(["yes","no"], n, p=[0.82,0.18]),
        "internet":    np.random.choice(["yes","no"], n, p=[0.78,0.22]),
        "romantic":    np.random.choice(["yes","no"], n, p=[0.35,0.65]),
        "famrel":      np.random.randint(1, 6, n),   # Family relationship 1-5
        "freetime":    np.random.randint(1, 6, n),
        "goout":       np.random.randint(1, 6, n),
        "Dalc":        np.random.randint(1, 6, n),   # Workday alcohol
        "Walc":        np.random.randint(1, 6, n),   # Weekend alcohol
        "health":      np.random.randint(1, 6, n),
        "absences":    np.random.randint(0, 33, n),
        "G1":          np.random.randint(0, 21, n),  # Period 1 grade
        "G2":          np.random.randint(0, 21, n),  # Period 2 grade
    }

    # G3 (final grade) correlates with G1, G2, studytime, failures
    df = pd.DataFrame(data)
    df["G3"] = (
        0.4 * df["G1"]
        + 0.4 * df["G2"]
        + df["studytime"] * 0.5
        - df["failures"] * 2
        + np.random.normal(0, 1.5, n)
    ).clip(0, 20).round().astype(int)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/student-mat.csv", sep=";", index=False)
    print(f"✓ Sample data generated: {n} students saved to data/student-mat.csv")
    return df


# ── 2. EXPLORE DATA ───────────────────────────────────────────
def explore_data(df):
    """Print key statistics about the dataset."""
    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\nColumn types:")
    print(df.dtypes.value_counts().to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    print("\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    else:
        print(missing[missing > 0].to_string())

    print("\nTarget variable (G3 — final grade) stats:")
    print(df["G3"].describe().round(2).to_string())

    print("\nGrade distribution:")
    bins = [0, 10, 14, 17, 20]
    labels = ["Fail (<10)", "Pass (10-13)", "Good (14-16)", "Excellent (17+)"]
    df["grade_category"] = pd.cut(df["G3"], bins=bins, labels=labels, include_lowest=True)
    print(df["grade_category"].value_counts().sort_index().to_string())
    df.drop(columns=["grade_category"], inplace=True)


# ── 3. VISUALIZE ──────────────────────────────────────────────
def visualize_data(df):
    """Generate 4 key plots for EDA."""
    os.makedirs("data", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Student Performance — Exploratory Data Analysis", fontsize=14, fontweight="bold")

    # Plot 1: Grade distribution
    sns.histplot(df["G3"], bins=21, kde=True, ax=axes[0, 0], color="#4f87ff")
    axes[0, 0].set_title("Final Grade (G3) Distribution")
    axes[0, 0].set_xlabel("Grade (0–20)")
    axes[0, 0].set_ylabel("Count")

    # Plot 2: Study time vs grade
    study_grade = df.groupby("studytime")["G3"].mean().reset_index()
    sns.barplot(data=study_grade, x="studytime", y="G3", ax=axes[0, 1], color="#22c9a0")
    axes[0, 1].set_title("Study Time vs Average Grade")
    axes[0, 1].set_xlabel("Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
    axes[0, 1].set_ylabel("Average G3")

    # Plot 3: Failures vs grade
    fail_grade = df.groupby("failures")["G3"].mean().reset_index()
    sns.barplot(data=fail_grade, x="failures", y="G3", ax=axes[1, 0], color="#ff6b6b")
    axes[1, 0].set_title("Past Failures vs Average Grade")
    axes[1, 0].set_xlabel("Number of Past Failures")
    axes[1, 0].set_ylabel("Average G3")

    # Plot 4: Correlation heatmap (numeric cols)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()[["G3"]].sort_values("G3", ascending=False).head(10)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1, 1],
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    axes[1, 1].set_title("Top Features Correlated with G3")

    plt.tight_layout()
    plt.savefig("data/eda_plots.png", dpi=120, bbox_inches="tight")
    print("\n✓ EDA plots saved to: data/eda_plots.png")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--sample" in sys.argv:
        df = generate_sample_data()
    else:
        df = load_data()
        if df is None:
            print("\nGenerating sample data instead...")
            df = generate_sample_data()

    explore_data(df)
    visualize_data(df)
    print("\n✓ Step 1 complete! Next: run python src/model.py")
