
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training examples
texts = [
    "I love this movie",
    "This is amazing",
    "I am very happy",
    "I hate this movie",
    "This is terrible",
    "I am very sad"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative"
]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train the AI model
model = MultinomialNB()
model.fit(X, labels)

# Test the model
sentence = input("Enter a sentence in English: ")
sentence_vector = vectorizer.transform([sentence])
prediction = model.predict(sentence_vector)

print("Sentiment:", prediction[0])
