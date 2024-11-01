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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


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
            "subcategory": LabelEncoder(),  # Changed from sub_category to subcategory
        }
        self.models = {
            "category": None,
            "subcategory": None
        }
        self.min_samples_per_class = min_samples_per_class

    def analyze_class_distribution(self, df):
        """Analyze and print class distribution information"""
        print("\nClass Distribution Analysis:", file=sys.stderr)
        
        for column in ['category', 'subcategory']:
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
        
        # Filter both category and subcategory
        for column in ['category', 'subcategory']:
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
        """Prepare the data for training"""
        # Ensure column names are correct
        df = df.rename(columns={'sub_category': 'subcategory'})
        
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
        for column in ['category', 'subcategory']:
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

    def train(self, df):
        """Train the model"""
        try:
            # Prepare data
            prepared_df = self.prepare_data(df)
            
            if len(prepared_df) < self.min_samples_per_class * 2:
                raise ValueError(f"Not enough samples for training. Need at least {self.min_samples_per_class * 2} samples.")
            
            # Split features
            X = prepared_df['processed_text']
            
            # Train separate models for category and subcategory
            for column in ['category', 'subcategory']:
                print(f"\nTraining {column} model...", file=sys.stderr)
                y = prepared_df[f'{column}_encoded']
                
                # Split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )
                
                # Build and train the model
                self.models[column] = self.build_model(self.label_encoders[column].classes_)
                self.models[column].fit(X_train, y_train)
                
                # Print evaluation metrics
                y_pred = self.models[column].predict(X_test)
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
        """Predict category and subcategory for new text"""
        try:
            # Preprocess the input text
            processed_text = self.preprocess_text(text)

            results = {}
            # Make predictions for each model
            for column in ['category', 'subcategory']:
                if self.models[column] is None:
                    raise ValueError(f"Model for {column} is not trained")
                
                # Get prediction and probabilities
                prediction = self.models[column].predict([processed_text])[0]
                probas = self.models[column].predict_proba([processed_text])[0]
                
                # Convert prediction to label
                results[column] = self.label_encoders[column].inverse_transform([prediction])[0]
                results[f"{column}_confidence"] = float(probas[prediction])

            return results

        except Exception as e:
            print(f"Error during prediction: {str(e)}", file=sys.stderr)
            raise

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
        print(f"Subcategory: {prediction['subcategory']} (confidence: {prediction['subcategory_confidence']:.2%})", file=sys.stderr)

    except Exception as e:
        print(f"Error in main: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
