# train_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import logging
import time
from datetime import datetime
from tqdm import tqdm
import sys

# Import the classifier from the previous file
from cybercrime_classifier import CybercrimeClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def print_progress(message):
    """Print message without interfering with progress bars"""
    print(f"\n{message}", file=sys.stderr)

def clean_dataframe(df):
    """Clean and prepare the dataframe before training"""
    print_progress("Cleaning dataset...")
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    total_steps = 3
    with tqdm(total=total_steps, desc="Cleaning Data") as pbar:
        # Handle null values in text column
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].fillna('')
        pbar.update(1)
        
        # Convert float/int values to string if any
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].astype(str)
        pbar.update(1)
        
        # Remove entries where text is empty or just whitespace
        df = df[df['crimeaditionalinfo'].str.strip() != '']
        df['category'] = df['category'].fillna('Unknown')
        df['sub_category'] = df['sub_category'].fillna('Unknown')
        pbar.update(1)
    
    # Log the cleaning results
    total_rows = len(df)
    null_counts = df.isnull().sum()
    print_progress(f"Dataset cleaned. Total rows after cleaning: {total_rows}")
    print_progress(f"Null value counts:\n{null_counts}")
    
    return df

def train_and_save_model():
    try:
        # Record start time
        start_time = time.time()
        
        print_progress("Starting model training process...")
        
        # Load training data
        with tqdm(total=1, desc="Loading Data") as pbar:
            train_df = pd.read_csv('data/train.csv')
            pbar.update(1)
        
        # Log initial data stats
        print_progress(f"Initial data shape: {train_df.shape}")
        print_progress("Initial null counts:")
        print_progress(str(train_df.isnull().sum()))
        
        # Clean the data
        train_df = clean_dataframe(train_df)
        
        # Initialize classifier
        classifier = CybercrimeClassifier()
        
        # Train the model with progress bar
        print_progress("Training model...")
        X_test, y_test = classifier.train(train_df)
        
        # Generate training metrics
        with tqdm(total=1, desc="Generating Metrics") as pbar:
            y_pred = classifier.model.predict(X_test)
            pbar.update(1)
        
        # Save metrics to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f'training_metrics_{timestamp}.txt'
        
        with tqdm(total=1, desc="Saving Results") as pbar:
            with open(metrics_file, 'w') as f:
                # Write data summary
                f.write("Data Summary:\n")
                f.write(f"Total samples: {len(train_df)}\n")
                f.write(f"Number of categories: {len(train_df['category'].unique())}\n")
                f.write(f"Number of sub-categories: {len(train_df['sub_category'].unique())}\n")
                f.write("\nCategory distribution:\n")
                f.write(train_df['category'].value_counts().to_string())
                f.write("\n\nSub-category distribution:\n")
                f.write(train_df['sub_category'].value_counts().to_string())
                f.write("\n\nClassification Reports:\n")
                
                # Write classification reports
                for i, column in enumerate(['Category', 'Sub-category']):
                    f.write(f"\n{column} Classification Report:\n")
                    report = classification_report(
                        y_test.iloc[:, i],
                        y_pred[:, i],
                        target_names=classifier.label_encoders[column.lower()].classes_
                    )
                    f.write(report)
                    f.write("\n")
                    
                    # Also log to console
                    print_progress(f"\n{column} Classification Report:\n{report}")
            
            # Save the model
            model_filename = f'cybercrime_classifier_{timestamp}.joblib'
            classifier.save_model(model_filename)
            pbar.update(1)
        
        # Calculate and log training time
        training_time = time.time() - start_time
        print_progress(f"Training completed in {training_time:.2f} seconds")
        print_progress(f"Model saved as: {model_filename}")
        print_progress(f"Metrics saved as: {metrics_file}")
        
        return model_filename, metrics_file
        
    except Exception as e:
        print_progress(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model_file, metrics_file = train_and_save_model()
        print("\nTraining completed successfully!")
        print(f"Model saved as: {model_file}")
        print(f"Metrics saved as: {metrics_file}")
    except Exception as e:
        print(f"Training failed: {str(e)}")