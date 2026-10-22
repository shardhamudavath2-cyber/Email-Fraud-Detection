from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Sample emails for demonstration
emails = [
    "You won a free prize click now",
    "Claim your cash reward immediately",
    "Verify your bank account now",
    "Congratulations you won a lottery",
    "Please find the class notes attached",
    "Our meeting is scheduled for tomorrow",
    "Your assignment is due on Friday",
    "Please review the project report",
    "The lecture starts at ten",
    "Can we discuss our project today"
]

labels = [
    "Fraud", "Fraud", "Fraud", "Fraud",
    "Safe", "Safe", "Safe", "Safe",
    "Safe", "Safe"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(emails, labels)

print("Email Fraud Detection System")
print("----------------------------")

while True:
    message = input(
        "\nEnter an email message (or type exit): "
    )

    if message.lower() == "exit":
        break

    prediction = model.predict([message])[0]
    print("Prediction:", prediction)
    print("Note: This is a basic demonstration model.")
