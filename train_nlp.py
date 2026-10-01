"""
NLP Model Training for Chatbot
Simple TF-IDF + Random Forest classifier
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data():
    """Load the chatbot dataset"""
    df = pd.read_csv('data/chatbot_data.csv')
    print(f"Loaded {len(df)} training samples")
    print(f"Intents: {df['intent'].nunique()}")
    return df

def train_intent_classifier(df):
    """Train intent classifier using TF-IDF and Random Forest"""
    print("\n" + "=" * 50)
    print("TRAINING INTENT CLASSIFIER")
    print("=" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['pattern'], df['intent'], test_size=0.2, random_state=42, stratify=df['intent']
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("\nCreating TF-IDF vectors...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF feature count: {X_train_tfidf.shape[1]}")
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42
    )
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(classifier, 'models/intent_classifier.pkl')
    print("\nModels saved to models/ directory")
    
    return tfidf_vectorizer, classifier, accuracy

def main():
    """Main training function"""
    print("=" * 50)
    print("NLP MODEL TRAINING")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Train classifier
    tfidf_vectorizer, classifier, accuracy = train_intent_classifier(df)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Final Accuracy: {accuracy:.4f}")
    print("\nModels saved:")
    print("- models/tfidf_vectorizer.pkl")
    print("- models/intent_classifier.pkl")

if __name__ == "__main__":
    main()
