import sys
import logging
from cybercrime_classifier import CybercrimeClassifier

def main():
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Load the pre-trained model
        try:
            classifier = CybercrimeClassifier.load_model('cybercrime_classifier.joblib')
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # Interactive prediction loop
        while True:
            # Prompt user for input
            print("\nEnter a cybercrime description (or 'quit' to exit):")
            user_input = input().strip()

            # Check for exit condition
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # Skip empty inputs
            if not user_input:
                print("Please enter a valid description.")
                continue

            try:
                # Predict category and subcategory
                prediction = classifier.predict(user_input)

                # Display results
                print("\n--- Prediction Results ---")
                print(f"Category: {prediction['category']} (Confidence: {prediction['category_confidence']:.2%})")
                print(f"Subcategory: {prediction['subcategory']} (Confidence: {prediction['subcategory_confidence']:.2%})")

            except Exception as e:
                print(f"Prediction error: {e}")

        print("\nThank you for using the Cybercrime Classifier!")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()