import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import os
import seaborn as sns

# Apply a nice style to the plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


def plot_rating_distribution(csv_file=None):
    """
    Step 1: Visualize the Data
    Reads the CSV and creates a bar chart of the star ratings.
    """
    if csv_file is None:
        csv_file = "../data/processed/cleaned_reviews.csv"
    
    print(f"--- Generating Rating Distribution from {csv_file} ---")

    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Trying raw reviews.csv...")
        csv_file = "../data/raw/reviews.csv"
        if not os.path.exists(csv_file):
            print("Error: No data file found. Skipping this graph.")
            return

    df = pd.read_csv(csv_file)

    # Check column name (stars or label)
    if 'stars' in df.columns:
        col = 'stars'
    elif 'label' in df.columns:
        col = 'label'
        # If label is 0-4, convert back to 1-5 for the chart
        if df['label'].max() < 5:
            df['stars_visual'] = df['label'] + 1
            col = 'stars_visual'
    else:
        print("Error: Could not find 'stars' or 'label' column.")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=col, data=df, palette="viridis")

    plt.title("Distribution of Star Ratings in Dataset", fontsize=16)
    plt.xlabel("Star Rating", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    # Add numbers on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    os.makedirs("../outputs/plots", exist_ok=True)
    plt.savefig("../outputs/plots/graph_rating_distribution.png")
    print("Saved: ../outputs/plots/graph_rating_distribution.png")
    plt.close()


def plot_accuracy_curve(predictions_dir=None):
    """
    Step 2: Visualize the Training
    Scans the checkpoints to find the history of accuracy improvement.
    """
    if predictions_dir is None:
        predictions_dir = "../outputs/checkpoints"
    
    print(f"\n--- Generating Accuracy Curve from {predictions_dir} ---")

    # Find all 'trainer_state.json' files in the subdirectories
    json_files = glob.glob(f"{predictions_dir}/**/trainer_state.json", recursive=True)

    if not json_files:
        print(f"No training logs found! Did you delete the '{predictions_dir}' folder?")
        print("Cannot generate accuracy graph automatically.")
        return

    # Use the most recently modified file (the latest run)
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"Reading logs from: {latest_file}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # Extract accuracy history
    history = data['log_history']

    epochs = []
    accuracies = []

    for entry in history:
        if 'eval_accuracy' in entry:
            epochs.append(entry['epoch'])
            accuracies.append(entry['eval_accuracy'])

    if not accuracies:
        print("Found logs, but no evaluation metrics. Did you set eval_strategy='epoch'?")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.title("Model Accuracy Improvement Over Time", fontsize=16)
    plt.xlabel("Epochs (Training Rounds)", fontsize=12)
    plt.ylabel("Accuracy (0-1.0)", fontsize=12)
    plt.ylim(0, 1.0)  # Set y-axis from 0% to 100%
    plt.grid(True)

    # Add text for the final accuracy
    plt.annotate(f'Final: {accuracies[-1]:.2%}',
                 (epochs[-1], accuracies[-1]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 fontweight='bold')

    os.makedirs("../outputs/plots", exist_ok=True)
    plt.savefig("../outputs/plots/graph_accuracy_curve.png")
    print("Saved: ../outputs/plots/graph_accuracy_curve.png")
    plt.close()


if __name__ == "__main__":
    plot_rating_distribution()
    plot_accuracy_curve()