import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Simple Dataset (Representing SMS Data)
data = {
    'text': [
        'Get 50% off on your next purchase!', 'Hey, are we still meeting for lunch?',
        'WINNER! You have won a prize of $1000.', 'Can you send me the notes for ADSA?',
        'URGENT: Your account has been compromised.', 'I will be late for the Java class.',
        'Free entry to the contest! Text WIN to 5050', 'Let us go for a movie tonight.'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# 2. Preprocessing: Convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

# 3. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Quick Test
def predict_sms(sms):
    sample = cv.transform([sms])
    return model.predict(sample)[0]

# Interactive Test
test_msg = "Claim your free gift card now!"
print(f"Message: {test_msg}")
print(f"Prediction: {predict_sms(test_msg)}")

# Show Accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
