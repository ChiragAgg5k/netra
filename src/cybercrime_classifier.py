import re
import sys

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
from tqdm.auto import tqdm
import logging


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
        self.label_encoders = {
            "category": LabelEncoder(),
            "sub_category": LabelEncoder(),
        }
        self.models = {
            "category": None,
            "sub_category": None
        }
        self.min_samples_per_class = min_samples_per_class

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

    def build_model(self, class_labels):
        """Create the classification pipeline"""
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])

    def train(self, train_df, test_df=None):
        """Train the model using separate train and test files"""
        try:
            # Prepare training data
            prepared_train_df = self.prepare_data(train_df)
            
            if len(prepared_train_df) < self.min_samples_per_class * 2:
                raise ValueError(f"Not enough samples for training. Need at least {self.min_samples_per_class * 2} samples.")
            
            # Split features for training
            X_train = prepared_train_df['processed_text']
            
            # Train separate models for category and sub_category
            for column in ['category', 'sub_category']:
                print(f"\nTraining {column} model...", file=sys.stderr)
                y_train = prepared_train_df[f'{column}_encoded']
                
                # Build and train the model
                self.models[column] = self.build_model(self.label_encoders[column].classes_)
                self.models[column].fit(X_train, y_train)
                
                # Evaluate on test data if provided
                if test_df is not None:
                    # Prepare test data with the same preprocessing
                    prepared_test_df = self.prepare_data(test_df)
                    X_test = prepared_test_df['processed_text']
                    y_test = prepared_test_df[f'{column}_encoded']
                    
                    # Make predictions
                    y_pred = self.models[column].predict(X_test)
                    
                    # Print evaluation metrics
                    print(f"\n{column.upper()} Classification Report:", file=sys.stderr)
                    print(classification_report(
                        y_test,
                        y_pred,
                        target_names=self.label_encoders[column].classes_,
                        zero_division=1
                    ))
            
            return True

        except Exception as e:
            print(f"Error during training: {str(e)}", file=sys.stderr)
            raise

    def predict(self, text):
        """Predict category and sub_category for new text with handling for unseen labels"""
        try:
            # Preprocess the input text
            processed_text = self.preprocess_text(text)

            results = {}
            # Make predictions for each model
            for column in ['category', 'sub_category']:
                if self.models[column] is None:
                    raise ValueError(f"Model for {column} is not trained")
                
                try:
                    # Get prediction and probabilities
                    prediction = self.models[column].predict([processed_text])[0]
                    probas = self.models[column].predict_proba([processed_text])[0]
                    
                    # Ensure prediction is within known labels
                    max_classes = len(self.label_encoders[column].classes_)
                    if prediction >= max_classes:
                        # !TODO: Handle this case
                        continue
                    
                    # Convert prediction to label
                    results[column] = self.label_encoders[column].classes_[prediction]
                    results[f"{column}_confidence"] = float(probas[prediction])
                
                except Exception as e:
                    logging.error(f"Error predicting {column}: {str(e)}")
                    # Fallback to "Unknown" category with 0 confidence
                    results[column] = "Unknown"
                    results[f"{column}_confidence"] = 0.0

            return results

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            # Return a complete result structure even in case of error
            return {
                "category": "Unknown",
                "category_confidence": 0.0,
                "sub_category": "Unknown",
                "sub_category_confidence": 0.0
            }

    def save_model(self, path):
        """Save the trained model and label encoders"""
        model_data = {
            "models": self.models,
            "label_encoders": self.label_encoders
        }
        joblib.dump(model_data, path)
        print(f"\nModel saved to {path}", file=sys.stderr)

    @classmethod
    def load_model(cls, path):
        """Load a trained model"""
        try:
            model_data = joblib.load(path)
            classifier = cls()
            classifier.models = model_data["models"]
            classifier.label_encoders = model_data["label_encoders"]
            return classifier
        except Exception as e:
            print(f"Error loading model: {str(e)}", file=sys.stderr)
            raise


def main():
    try:
        # Load data
        print("Loading data...", file=sys.stderr)
        df = pd.read_csv("data/train.csv")
        print(f"Loaded {len(df)} records", file=sys.stderr)

        # Initialize and train classifier
        classifier = CybercrimeClassifier(min_samples_per_class=5)  # Adjust this value as needed
        classifier.train(df)

        # Save the model
        classifier.save_model("cybercrime_classifier.joblib")

        # Example prediction
        sample_text = """I had continue received random calls and abusive messages in my whatsapp 
        Someone added my number in a unknown facebook group name with Only Girls and still getting 
        calls from unknown numbers"""

        prediction = classifier.predict(sample_text)
        print("\nSample Prediction:", file=sys.stderr)
        print("Sample Text:", sample_text, file=sys.stderr)
        print(f"Category: {prediction['category']} (confidence: {prediction['category_confidence']:.2%})", file=sys.stderr)
        print(f"sub_category: {prediction['sub_category']} (confidence: {prediction['sub_category_confidence']:.2%})", file=sys.stderr)

    except Exception as e:
        print(f"Error in main: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
