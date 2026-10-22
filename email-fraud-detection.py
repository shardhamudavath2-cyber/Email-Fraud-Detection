import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/spambase/spambase.data"
)

MODEL_FILE = "email_spam_model.pkl"

# Column names for the 57 numerical email features
FEATURES = [
    "word_freq_make", "word_freq_address", "word_freq_all",
    "word_freq_3d", "word_freq_our", "word_freq_over",
    "word_freq_remove", "word_freq_internet", "word_freq_order",
    "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email",
    "word_freq_you", "word_freq_credit", "word_freq_your",
    "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george",
    "word_freq_650", "word_freq_lab", "word_freq_labs",
    "word_freq_telnet", "word_freq_857", "word_freq_data",
    "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm",
    "word_freq_direct", "word_freq_cs", "word_freq_meeting",
    "word_freq_original", "word_freq_project", "word_freq_re",
    "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_semicolon", "char_freq_parenthesis",
    "char_freq_bracket", "char_freq_bang", "char_freq_dollar",
    "char_freq_hash", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total"
]

def main():
    print("EMAIL SPAM DETECTION")
    print("-" * 30)

    # Load the labelled dataset
    df = pd.read_csv(DATA_URL, header=None, names=FEATURES + ["label"])
    X = df[FEATURES]
    y = df["label"]

    # Use the same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Load the trained model
    model = joblib.load(MODEL_FILE)

    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        target_names=["Non-Spam", "Spam"]
    ))

    # Try one record from the test dataset
    while True:
        choice = input(
            "\nEnter a test record number (0-920), or q to quit: "
        ).strip()

        if choice.lower() == "q":
            print("Goodbye!")
            break

        if not choice.isdigit() or not 0 <= int(choice) < len(X_test):
            print("Please enter a number from 0 to 920, or q.")
            continue

        index = int(choice)
        sample = X_test.iloc[[index]]
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        print("Prediction:", "SPAM" if prediction == 1 else "NON-SPAM")
        print(f"Estimated spam probability: {probability * 100:.2f}%")
        print("Actual label:", "SPAM" if y_test.iloc[index] == 1 else "NON-SPAM")

if _name_ == "_main_":
    main()

