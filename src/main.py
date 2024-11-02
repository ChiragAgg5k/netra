import logging
import sys

from cybercrime_classifier import CybercrimeClassifier


def main():
    """Main function to run the Cybercrime Classifier."""
    try:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        try:
            classifier = CybercrimeClassifier.load_model("cybercrime_classifier.joblib")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        while True:
            print("\nEnter a cybercrime description (or 'quit' to exit):")
            user_input = input().strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                print("Please enter a valid description.")
                continue

            try:
                prediction = classifier.predict(user_input)

                print("\n--- Prediction Results ---")
                print(
                    f"Category: {prediction['category']} (Confidence: {prediction['category_confidence']:.2%})"
                )
                print(
                    f"sub_category: {prediction['sub_category']} (Confidence: {prediction['sub_category_confidence']:.2%})"
                )

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
