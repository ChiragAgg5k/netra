# train.py

import logging
import sys
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from cybercrime_classifier import CybercrimeClassifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def print_progress(message):
    """Print message without interfering with progress bars."""
    print(f"\n{message}", file=sys.stderr)

def clean_dataframe(df):
    """Clean and prepare the dataframe before training.

    Args:
        df (DataFrame): The dataframe to clean.

    Returns:
        DataFrame: The cleaned dataframe.
    """
    df = df.copy()
    
    total_steps = 3
    with tqdm(total=total_steps, desc="Cleaning Data") as pbar:
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].fillna('')
        pbar.update(1)
        
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].astype(str)
        pbar.update(1)
        
        df = df[df['crimeaditionalinfo'].str.strip() != '']
        df['category'] = df['category'].fillna('Unknown')
        df['sub_category'] = df['sub_category'].fillna('Unknown')
        pbar.update(1)
    
    total_rows = len(df)
    null_counts = df.isnull().sum()
    print_progress(f"Dataset cleaned. Total rows after cleaning: {total_rows}")
    print_progress(f"Null value counts:\n{null_counts}")
    
    return df

def train_and_save_model():
    """Train the CybercrimeClassifier model and save it along with metrics.

    Returns:
        tuple: The model filename and metrics filename.
    """
    try:
        start_time = time.time()
        
        print_progress("Starting model training process...")
        
        with tqdm(total=1, desc="Loading Data") as pbar:
            train_df = pd.read_csv('data/train.csv')
            pbar.update(1)
        
        print_progress(f"Initial data shape: {train_df.shape}")
        print_progress("Initial null counts:")
        print_progress(str(train_df.isnull().sum()))
        
        train_df = clean_dataframe(train_df)
        
        classifier = CybercrimeClassifier(min_samples_per_class=5)
        
        print_progress("Training model...")
        classifier.train(train_df)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f'training_metrics_{timestamp}.txt'
        
        with open(metrics_file, 'w') as f:
            f.write("Data Summary:\n")
            f.write(f"Total samples: {len(train_df)}\n")
            f.write(f"Number of categories: {len(train_df['category'].unique())}\n")
            f.write(f"Number of sub-categories: {len(train_df['sub_category'].unique())}\n")
            f.write("\nCategory distribution:\n")
            f.write(train_df['category'].value_counts().to_string())
            f.write("\n\nSub-category distribution:\n")
            f.write(train_df['sub_category'].value_counts().to_string())
        
        model_filename = f'cybercrime_classifier_{timestamp}.joblib'
        classifier.save_model(model_filename)
        
        training_time = time.time() - start_time
        print_progress(f"Training completed in {training_time:.2f} seconds")
        print_progress(f"Model saved as: {model_filename}")
        print_progress(f"Metrics saved as: {metrics_file}")
        
        return model_filename, metrics_file
        
    except Exception as e:
        print_progress(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    """Entry point for the training script."""
    try:
        model_file, metrics_file = train_and_save_model()
        print("\nTraining completed successfully!")
        print(f"Model saved as: {model_file}")
        print(f"Metrics saved as: {metrics_file}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
