import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Path to your CSV file
file_path = 'C:/Users/bhava/Downloads/python/spam.csv'  # Update the path as needed

# Initialize an empty list to store the data
data = []

# Open and read the CSV file
with open(file_path, newline='', encoding='latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    rows = list(csvreader)  # Read all rows into a list
    
    # Check if the file is empty
    if not rows:
        print("Error: The CSV file is empty.")
    else:
        print(f"Number of rows read: {len(rows)}")  # Debugging line
        
        # Skip the header and process the data
        for row in rows[1:]:  # Skip the first row (header)
            if row:  # Skip empty rows
                label = 0 if row[0] == 'ham' else 1  # Map 'ham' to 0 and 'spam' to 1
                message = row[1]
                data.append([label, message])

# If there is no data, handle the error
if not data:
    print("Error: No data found in the CSV file.")
else:
    # Proceed with the rest of the code for processing and classification
    # Data Preprocessing
    X = [row[1] for row in data]  # Messages
    y = [row[0] for row in data]  # Labels (0 = ham, 1 = spam)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model training using Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Making predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model and vectorizer for future use
    joblib.dump(model, "spam_classifier_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
