import re
import sys
import logging
import numpy as np
from typing import Dict, Any

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("testing.log"), logging.StreamHandler()],
)

class CybercrimeClassifier:
    def __init__(self, min_samples_per_class=2):
        # Download required NLTK resources
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Add domain-specific stop words
        self.stop_words.update(['please', 'help', 'thank', 'thanks', 'sir', 'madam', 'kindly'])
        
        self.label_encoders = {
            "category": LabelEncoder(),
            "sub_category": LabelEncoder(),
        }
        self.models = {
            "category": None,
            "sub_category": None
        }
        self.min_samples_per_class = min_samples_per_class
        self.vectorizer_params = {
            'max_features': 10000,  # Increased from 5000
            'ngram_range': (1, 3),  # Added trigrams
            'min_df': 2,
            'max_df': 0.95,
            'analyzer': 'word',
            'token_pattern': r'\b\w+\b',  # Matches whole words
        }

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with domain-specific cleaning"""
        try:
            if not isinstance(text, str):
                text = str(text)

            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove phone numbers (various formats)
            text = re.sub(r'\+?\d{10,}|\+?\d{3}[-\s]?\d{3}[-\s]?\d{4}', '', text)
            
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Keep important punctuation that might indicate sentiment or emphasis
            text = re.sub(r'[^a-zA-Z\s!?.]', '', text)

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize, keep tokens longer than 2 characters
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]

            # Add bigrams and trigrams for important phrases
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens)-2)]
            
            processed_text = ' '.join(tokens + bigrams + trigrams)
            
            # Return empty string if processed text is too short
            if len(processed_text.split()) < 3:
                return ""
                
            return processed_text

        except Exception as e:
            logging.error(f"Error preprocessing text: {str(e)}")
            return ""

    def build_model(self, class_labels: np.ndarray) -> Pipeline:
        """Create an improved classification pipeline with hyperparameter tuning"""
        # Calculate class weights
        n_samples = len(class_labels)
        n_classes = len(np.unique(class_labels))
        
        # If severe class imbalance, adjust class weights
        if n_classes > 1:
            class_weights = dict(zip(
                np.unique(class_labels),
                n_samples / (n_classes * np.bincount(class_labels))
            ))
        else:
            class_weights = None

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**self.vectorizer_params)),
            ("classifier", RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=20,  # Added max_depth to prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight=class_weights,
                bootstrap=True
            ))
        ])

        # Define parameters for grid search
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 2), (1, 3)]
        }

        return GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='f1_weighted'
        )

    def analyze_class_distribution(self, df):
        """Analyze and print class distribution information"""
        print("\nClass Distribution Analysis:", file=sys.stderr)
        
        for column in ['category', 'sub_category']:
            counts = df[column].value_counts()
            print(f"\n{column.upper()} Distribution:", file=sys.stderr)
            print(f"Total unique classes: {len(counts)}", file=sys.stderr)
            print(f"Classes with only one sample: {sum(counts == 1)}", file=sys.stderr)
            print("\nTop 5 most common classes:", file=sys.stderr)
            print(counts.head().to_string(), file=sys.stderr)
            print("\nClasses with less than minimum samples:", file=sys.stderr)
            print(counts[counts < self.min_samples_per_class].to_string(), file=sys.stderr)
        
        return counts

    def filter_rare_classes(self, df):
        """Filter out classes with too few samples"""
        print("\nFiltering rare classes...", file=sys.stderr)
        
        original_len = len(df)
        
        # Filter both category and sub_category
        for column in ['category', 'sub_category']:
            counts = df[column].value_counts()
            valid_classes = counts[counts >= self.min_samples_per_class].index
            df = df[df[column].isin(valid_classes)]
        
        filtered_len = len(df)
        print(f"Removed {original_len - filtered_len} samples with rare classes", file=sys.stderr)
        print(f"Remaining samples: {filtered_len}", file=sys.stderr)
        
        if filtered_len == 0:
            raise ValueError("No samples remaining after filtering rare classes. Consider lowering min_samples_per_class.")
        
        return df

    def preprocess_text(self, text):
        """Clean and preprocess the text data"""
        try:
            # Convert to string if not already
            if not isinstance(text, str):
                text = str(text)

            # Convert to lowercase
            text = text.lower()

            # Remove special characters and numbers
            text = re.sub(r"[^a-zA-Z\s]", "", text)

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]

            return " ".join(tokens)

        except Exception as e:
            print(f"Error preprocessing text: {str(e)}", file=sys.stderr)
            return ""  # Return empty string in case of error

    def prepare_data(self, df):
        """Prepare the data for training with additional validation"""
        
        # Add Unknown category for handling unseen labels
        for column in ['category', 'sub_category']:
            if 'Unknown' not in df[column].unique():
                # Add a single example of Unknown category
                unknown_row = df.iloc[0].copy()
                unknown_row[column] = 'Unknown'
                unknown_row['crimeaditionalinfo'] = 'unknown case'
                df = pd.concat([df, pd.DataFrame([unknown_row])], ignore_index=True)
        
        # Analyze initial class distribution
        self.analyze_class_distribution(df)
        
        # Filter rare classes
        df = self.filter_rare_classes(df)
        
        # Analyze class distribution after filtering
        print("\nClass distribution after filtering:", file=sys.stderr)
        self.analyze_class_distribution(df)
        
        # Preprocess the text data
        print("\nPreprocessing text data...", file=sys.stderr)
        df['processed_text'] = [
            self.preprocess_text(text) 
            for text in tqdm(df['crimeaditionalinfo'], desc="Preprocessing Text")
        ]
        
        # Remove empty processed texts
        df = df[df['processed_text'].str.len() > 0]
        print(f"Samples after removing empty processed texts: {len(df)}", file=sys.stderr)
        
        # Encode labels
        print("\nEncoding labels...", file=sys.stderr)
        for column in ['category', 'sub_category']:
            df[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column])
            print(f"Number of unique {column}s: {len(self.label_encoders[column].classes_)}", file=sys.stderr)
        
        return df

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> bool:
        """Enhanced training with cross-validation and error analysis"""
        try:
            # Prepare training data
            prepared_train_df = self.prepare_data(train_df)
            
            if len(prepared_train_df) < self.min_samples_per_class * 2:
                raise ValueError(f"Not enough samples for training. Need at least {self.min_samples_per_class * 2} samples.")
            
            # Split features for training
            X = prepared_train_df['processed_text']
            
            # If no test set provided, create one
            if test_df is None:
                # Stratified split to maintain class distribution
                X_train, X_val, y_train_dict, y_val_dict = {}, {}, {}, {}
                
                for column in ['category', 'sub_category']:
                    y = prepared_train_df[f'{column}_encoded']
                    X_train[column], X_val[column], y_train_dict[column], y_val_dict[column] = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
            else:
                prepared_test_df = self.prepare_data(test_df)
                X_val = {
                    'category': prepared_test_df['processed_text'],
                    'sub_category': prepared_test_df['processed_text']
                }
                y_val_dict = {
                    'category': prepared_test_df['category_encoded'],
                    'sub_category': prepared_test_df['sub_category_encoded']
                }
                X_train = {'category': X, 'sub_category': X}
                y_train_dict = {
                    'category': prepared_train_df['category_encoded'],
                    'sub_category': prepared_train_df['sub_category_encoded']
                }

            # Train and evaluate models for each target
            for column in ['category', 'sub_category']:
                print(f"\nTraining {column} model...", file=sys.stderr)
                
                # Build and train the model with grid search
                self.models[column] = self.build_model(y_train_dict[column])
                self.models[column].fit(X_train[column], y_train_dict[column])
                
                # Print best parameters
                print(f"\nBest parameters for {column}:", file=sys.stderr)
                print(self.models[column].best_params_, file=sys.stderr)
                
                # Make predictions and evaluate
                y_pred = self.models[column].predict(X_val[column])
                
                # Print detailed evaluation metrics
                print(f"\n{column.upper()} Classification Report:", file=sys.stderr)
                print(classification_report(
                    y_val_dict[column],
                    y_pred,
                    target_names=self.label_encoders[column].classes_,
                    zero_division=1
                ))
                
                # Error analysis
                self._perform_error_analysis(
                    X_val[column],
                    y_val_dict[column],
                    y_pred,
                    self.label_encoders[column].classes_,
                    column
                )
            
            return True

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def _perform_error_analysis(self, X_val, y_true, y_pred, class_names, column):
        """Analyze prediction errors to identify patterns"""
        # Convert encoded labels back to original names
        y_true_names = [class_names[i] for i in y_true]
        y_pred_names = [class_names[i] for i in y_pred]
        
        # Find misclassified examples
        errors = [(true, pred, text) for true, pred, text in zip(y_true_names, y_pred_names, X_val) if true != pred]
        
        if errors:
            print(f"\nError Analysis for {column}:", file=sys.stderr)
            print(f"Total errors: {len(errors)}", file=sys.stderr)
            
            # Analyze common misclassifications
            misclass_pairs = [(true, pred) for true, pred, _ in errors]
            common_errors = pd.DataFrame(misclass_pairs, columns=['True', 'Predicted']).value_counts().head()
            
            print("\nMost common misclassifications:", file=sys.stderr)
            print(common_errors.to_string(), file=sys.stderr)

    def predict(self, text: str) -> Dict[str, Any]:
        """Enhanced prediction with confidence thresholds and error handling"""
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return self._get_unknown_prediction()

            results = {}
            for column in ['category', 'sub_category']:
                if self.models[column] is None:
                    raise ValueError(f"Model for {column} is not trained")
                
                try:
                    # Get prediction probabilities
                    probas = self.models[column].predict_proba([processed_text])[0]
                    max_proba = max(probas)
                    prediction = self.models[column].predict([processed_text])[0]
                    
                    # Use confidence threshold
                    if max_proba < 0.3:  # Adjusted confidence threshold
                        results[column] = "Unknown"
                        results[f"{column}_confidence"] = 0.0
                    else:
                        results[column] = self.label_encoders[column].classes_[prediction]
                        results[f"{column}_confidence"] = float(max_proba)
                
                except Exception as e:
                    logging.error(f"Error predicting {column}: {str(e)}")
                    results.update(self._get_unknown_prediction(column))

            return results

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return self._get_unknown_prediction()

    def _get_unknown_prediction(self, column: str = None) -> Dict[str, Any]:
        """Helper method to return unknown prediction"""
        if column:
            return {
                column: "Unknown",
                f"{column}_confidence": 0.0
            }
        return {
            "category": "Unknown",
            "category_confidence": 0.0,
            "sub_category": "Unknown",
            "sub_category_confidence": 0.0
        }