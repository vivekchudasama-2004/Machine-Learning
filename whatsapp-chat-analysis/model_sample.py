import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Step 1: Load dataset
df = pd.read_csv("sentiment_data.csv")  # Replace with actual filename

# Step 2: Clean and preprocess labels
df['sentiment'] = df['sentiment'].str.strip().str.lower()

# Step 3: Filter only valid sentiments
valid_labels = ['negative', 'neutral', 'positive']
df = df[df['sentiment'].isin(valid_labels)]

# Step 4: Balance the dataset
positive = df[df['sentiment'] == 'positive']
neutral = df[df['sentiment'] == 'neutral']
negative = df[df['sentiment'] == 'negative']

# Upsample minority classes to match neutral
positive_upsampled = resample(positive, replace=True, n_samples=len(neutral), random_state=42)
negative_upsampled = resample(negative, replace=True, n_samples=len(neutral), random_state=42)

# Combine balanced data
df_balanced = pd.concat([neutral, positive_upsampled, negative_upsampled])

# Step 5: Vectorize messages using TF-IDF with bigrams
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df_balanced['message'])
y = df_balanced['sentiment']

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model (you can also try RandomForest or GradientBoosting)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_labels, yticklabels=valid_labels, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Step 9: Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved.")
