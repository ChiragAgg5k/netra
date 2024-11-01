import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from cybercrime_classifier import CybercrimeClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("testing.log"), logging.StreamHandler()],
)

def plot_confusion_matrix(cm, classes, title, filename):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def test_model(model_path):
    try:
        logging.info("Starting model testing process...")

        # Load the model
        logging.info(f"Loading model from {model_path}...")
        classifier = CybercrimeClassifier.load_model(model_path)

        # Load test data
        logging.info("Loading test data...")
        test_df = pd.read_csv("data/test.csv")

        test_df = test_df.rename(columns={
            'sub_category': 'sub_category', 
            'crimeaditionalinfo': 'processed_text'
        })

        # Preprocess test data
        logging.info("Preprocessing test data...")
        test_df = classifier.prepare_data(test_df)

        # Get predictions and true values
        logging.info("Making predictions...")
        X_test = test_df["processed_text"]
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize results dictionary
        results = {"timestamp": timestamp, "model_path": model_path, "metrics": {}}

        # Generate and save detailed metrics
        metrics_file = f"test_metrics_{timestamp}.txt"
        
        with open(metrics_file, "w") as f:
            for column in ['category', 'sub_category']:
                # Get predictions for current model with tqdm progress bar
                logging.info(f"Generating predictions for {column}...")
                predictions = []
                probabilities = []
                
                for text in tqdm(X_test, desc=f"Predicting {column}", unit="sample"):
                    pred = classifier.predict(text)
                    predictions.append(pred[column])
                    probabilities.append(pred[f"{column}_confidence"])
                
                # Get true labels
                true_labels = test_df[column]
                
                # Classification report
                report = classification_report(
                    true_labels,
                    predictions,
                    output_dict=True
                )

                # Save to results dictionary
                results["metrics"][column] = report

                # Write to file
                f.write(f"\n{column.upper()} Classification Report:\n")
                f.write(classification_report(true_labels, predictions))

                # Generate confusion matrix
                unique_labels = sorted(set(list(true_labels) + list(predictions)))
                cm = confusion_matrix(
                    true_labels, 
                    predictions,
                    labels=unique_labels
                )

                # Plot and save confusion matrix
                plot_confusion_matrix(
                    cm,
                    unique_labels,
                    f"{column.title()} Confusion Matrix",
                    f"confusion_matrix_{column.lower()}_{timestamp}.png",
                )

                # Log metrics
                logging.info(f"\n{column.upper()} Classification Report:\n{report}")

        # Save results to JSON
        results_file = f"test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        # Generate example predictions
        logging.info("\nGenerating example predictions...")
        n_samples = min(5, len(test_df))
        sample_indices = np.random.choice(len(test_df), n_samples, replace=False)

        examples_file = f"example_predictions_{timestamp}.txt"
        with open(examples_file, "w") as f:
            f.write("Example Predictions:\n\n")
            for idx in tqdm(sample_indices, desc="Generating Example Predictions", unit="sample"):
                text = test_df.iloc[idx]["crimeaditionalinfo"]
                true_cat = test_df.iloc[idx]["category"]
                true_subcat = test_df.iloc[idx]["sub_category"]

                prediction = classifier.predict(text)

                f.write("-" * 80 + "\n")
                f.write(f"Text: {text}\n")
                f.write(f"True Category: {true_cat}\n")
                f.write(f"Predicted Category: {prediction['category']}\n")
                f.write(f"Category Confidence: {prediction['category_confidence']:.2%}\n")
                f.write(f"True sub_category: {true_subcat}\n")
                f.write(f"Predicted sub_category: {prediction['sub_category']}\n")
                f.write(f"sub_category Confidence: {prediction['sub_category_confidence']:.2%}\n\n")

        logging.info("Testing completed successfully")
        logging.info(f"Metrics saved to: {metrics_file}")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Example predictions saved to: {examples_file}")

        return metrics_file, results_file, examples_file

    except Exception as e:
        logging.error(f"Error during testing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Specify the path to your trained model
        model_path = "cybercrime_classifier.joblib"
        metrics_file, results_file, examples_file = test_model(model_path)

        print("\nTesting completed successfully!")
        print(f"Metrics saved as: {metrics_file}")
        print(f"Results saved as: {results_file}")
        print(f"Example predictions saved as: {examples_file}")
    except Exception as e:
        print(f"Testing failed: {str(e)}")