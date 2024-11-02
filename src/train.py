import logging
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

from cybercrime_classifier import CybercrimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyzes dataset characteristics and quality."""
    
    @staticmethod
    def analyze_text_length(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze text length distribution."""
        text_lengths = df['crimeaditionalinfo'].str.len()
        return {
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'mean_length': float(text_lengths.mean()),
            'median_length': float(text_lengths.median()),
            'std_length': float(text_lengths.std())
        }
    
    @staticmethod
    def analyze_class_distribution(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Analyze class distribution for categories and sub-categories."""
        return {
            'category': df['category'].value_counts().to_dict(),
            'sub_category': df['sub_category'].value_counts().to_dict()
        }
    
    @staticmethod
    def analyze_class_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze class overlap between train and test sets."""
        train_categories = set(train_df['category'].unique())
        test_categories = set(test_df['category'].unique())
        train_subcategories = set(train_df['sub_category'].unique())
        test_subcategories = set(test_df['sub_category'].unique())
        
        return {
            'category': {
                'train_only': list(train_categories - test_categories),
                'test_only': list(test_categories - train_categories),
                'common': list(train_categories & test_categories)
            },
            'sub_category': {
                'train_only': list(train_subcategories - test_subcategories),
                'test_only': list(test_subcategories - train_subcategories),
                'common': list(train_subcategories & test_subcategories)
            }
        }
    
    @staticmethod
    def detect_potential_issues(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential data quality issues."""
        issues = {
            'duplicate_texts': int(df['crimeaditionalinfo'].duplicated().sum()),
            'very_short_texts': int((df['crimeaditionalinfo'].str.len() < 20).sum()),
            'very_long_texts': int((df['crimeaditionalinfo'].str.len() > 1000).sum()),
            'potential_noise': int((df['crimeaditionalinfo'].str.contains(r'[^\w\s.,!?@#$%&*()]')).sum())
        }
        return issues

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and filter data, ensuring minimum samples per class."""
        try:
            prepared_df = df.copy()
            
            # Preprocess text
            prepared_df['processed_text'] = prepared_df['crimeaditionalinfo'].apply(self.preprocess_text)
            
            # Remove rows where preprocessing resulted in empty strings
            prepared_df = prepared_df[prepared_df['processed_text'].str.len() > 0]
            
            # Filter out classes with insufficient samples
            for column in ['category', 'sub_category']:
                # Get class counts
                class_counts = prepared_df[column].value_counts()
                
                # Get valid classes (those with enough samples)
                valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
                
                if len(valid_classes) == 0:
                    raise ValueError(f"No classes have the minimum required {self.min_samples_per_class} samples")
                
                # Filter dataframe to keep only valid classes
                prepared_df = prepared_df[prepared_df[column].isin(valid_classes)]
                
                # Fit label encoder on valid classes
                self.label_encoders[column].fit(valid_classes)
                
                # Encode labels
                prepared_df[f'{column}_encoded'] = self.label_encoders[column].transform(prepared_df[column])
                
                # Log removed classes
                removed_classes = set(class_counts.index) - set(valid_classes)
                if removed_classes:
                    logging.warning(f"Removed {len(removed_classes)} {column} classes with fewer than "
                                f"{self.min_samples_per_class} samples: {removed_classes}")
            
            return prepared_df

        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise

def clean_dataframe(df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, list]:
    """Enhanced data cleaning with quality checks and reporting."""
    df = df.copy()
    
    cleaning_steps = [
        ('Initial shape', lambda x: x, 'Initial dataset loaded'),
        ('Remove empty texts', lambda x: x[x['crimeaditionalinfo'].str.strip() != ''], 'Removed empty texts'),
        ('Fill missing values', lambda x: x.fillna({'category': 'Unknown', 'sub_category': 'Unknown', 'crimeaditionalinfo': ''}), 'Filled missing values'),
        ('Convert to string', lambda x: x.assign(crimeaditionalinfo=x['crimeaditionalinfo'].astype(str)), 'Converted text to string'),
        ('Clean special characters', lambda x: x.assign(
            crimeaditionalinfo=x['crimeaditionalinfo'].str.replace(r'[^\w\s.,!?@#$%&*()]', ' ', regex=True)
        ), 'Cleaned special characters')
    ]
    
    # Only remove duplicates and very short texts from training data
    if is_training:
        cleaning_steps.extend([
            ('Remove duplicates', lambda x: x.drop_duplicates(subset='crimeaditionalinfo'), 'Removed duplicate texts'),
            ('Remove very short texts', lambda x: x[x['crimeaditionalinfo'].str.len() >= 20], 'Removed very short texts')
        ])
    
    cleaning_report = []
    
    for step_name, step_func, step_desc in tqdm(cleaning_steps, desc="Cleaning Data"):
        initial_shape = df.shape[0]
        df = step_func(df)
        final_shape = df.shape[0]
        
        cleaning_report.append({
            'step': step_name,
            'description': step_desc,
            'rows_before': initial_shape,
            'rows_after': final_shape,
            'rows_removed': initial_shape - final_shape
        })
    
    return df, cleaning_report

def save_metrics(metrics: Dict[str, Any], filename: str):
    """Save metrics with proper formatting."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def train_and_save_model(
    train_path: str = 'data/train.csv',
    test_path: str = 'data/test.csv',
    min_samples_per_class: int = 5,
    output_dir: str = 'output'
) -> Tuple[str, str]:
    """Enhanced training pipeline with better error handling and class filtering."""
    try:
        start_time = time.time()
        logger.info("Starting enhanced training pipeline...")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load data
        logger.info("Loading datasets...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Analyze initial class distribution
        logger.info("Analyzing class distribution...")
        for column in ['category', 'sub_category']:
            class_counts = train_df[column].value_counts()
            logger.info(f"\n{column} distribution:")
            logger.info(f"Classes with 1 sample: {sum(class_counts == 1)}")
            logger.info(f"Classes with 2-4 samples: {sum((class_counts >= 2) & (class_counts < 5))}")
            logger.info(f"Classes with 5+ samples: {sum(class_counts >= 5)}")
        
        # Clean data
        train_df_cleaned, train_cleaning_report = clean_dataframe(train_df, is_training=True)
        test_df_cleaned, test_cleaning_report = clean_dataframe(test_df, is_training=False)
        
        # Initialize classifier with minimum samples requirement
        classifier = CybercrimeClassifier(min_samples_per_class=min_samples_per_class)
        
        # Prepare and filter data
        logger.info(f"Preparing data with minimum {min_samples_per_class} samples per class...")
        try:
            train_df_filtered = classifier.prepare_data(train_df_cleaned)
            test_df_filtered = classifier.prepare_data(test_df_cleaned)
            
            logger.info(f"Training data shape after filtering: {train_df_filtered.shape}")
            logger.info(f"Test data shape after filtering: {test_df_filtered.shape}")
            
            # Create validation set
            train_final, val_df = train_test_split(
                train_df_filtered,
                test_size=0.1,
                random_state=42
            )
            
            # Train model
            logger.info("Training model...")
            classifier.train(train_final, val_df)
            
            # Save model and metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = output_dir / f'cybercrime_classifier_{timestamp}.joblib'
            metrics_filename = output_dir / f'training_metrics_{timestamp}.json'
            
            joblib.dump(classifier, model_filename)
            
            # Save summary metrics
            metrics = {
                'timestamp': timestamp,
                'training_time': time.time() - start_time,
                'data_shapes': {
                    'initial_train': train_df.shape,
                    'filtered_train': train_df_filtered.shape,
                    'validation': val_df.shape,
                    'test': test_df_filtered.shape
                },
                'class_counts': {
                    'category': train_df_filtered['category'].value_counts().to_dict(),
                    'sub_category': train_df_filtered['sub_category'].value_counts().to_dict()
                }
            }
            
            with open(metrics_filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return str(model_filename), str(metrics_filename)
            
        except ValueError as ve:
            logger.error(f"Data preparation failed: {str(ve)}")
            raise
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model_file, metrics_file = train_and_save_model()
        print("\nTraining completed successfully!")
        print(f"Model saved as: {model_file}")
        print(f"Metrics saved as: {metrics_file}")
        
        # Load and display key metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            print("\nKey Performance Metrics:")
            print(f"Training Time: {metrics['training_time']:.2f} seconds")
            print("\nModel Performance:")
            print(json.dumps(metrics['model_performance'], indent=2))
            
            # Print class overlap warnings if any
            overlap = metrics['data_analysis']['initial']['class_overlap']
            if overlap['category']['test_only'] or overlap['sub_category']['test_only']:
                print("\nWarning: Test set contains classes not seen during training:")
                if overlap['category']['test_only']:
                    print(f"Categories: {', '.join(overlap['category']['test_only'])}")
                if overlap['sub_category']['test_only']:
                    print(f"Sub-categories: {', '.join(overlap['sub_category']['test_only'])}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)