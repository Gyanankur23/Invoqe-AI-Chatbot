"""
Chatbot Engine for Intent Classification and Response Generation
"""

import json
import random
import joblib
import numpy as np

class ChatbotEngine:
    """Main chatbot engine for intent classification and response generation"""
    
    def __init__(self):
        """Load trained models and intents"""
        try:
            self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            self.classifier = joblib.load('models/intent_classifier.pkl')
            
            with open('models/intents.json', 'r') as f:
                self.intents_data = json.load(f)
            
            print("Chatbot engine loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run train_nlp.py first to train the models")
            raise
    
    def classify_intent(self, user_input):
        """Classify the intent of user input"""
        # Preprocess input (simple cleaning)
        user_input = user_input.lower()
        user_input = ' '.join([c for c in user_input if c.isalnum() or c.isspace()])
        
        # Transform using TF-IDF
        input_tfidf = self.vectorizer.transform([user_input])
        
        # Predict intent
        intent = self.classifier.predict(input_tfidf)[0]
        
        # Get probability/confidence
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(input_tfidf)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0
        
        return intent, confidence
    
    def get_response(self, intent):
        """Get a response for the classified intent"""
        # Find intent in intents data
        for intent_data in self.intents_data['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                return random.choice(responses)
        
        # Default response if intent not found
        return "I'm not sure how to help with that. Could you please rephrase your question?"
    
    def chat(self, user_input):
        """Process user input and return response"""
        # Classify intent
        intent, confidence = self.classify_intent(user_input)
        
        # Get response
        response = self.get_response(intent)
        
        return {
            'intent': intent,
            'confidence': float(confidence),
            'response': response
        }
    
    def get_all_intents(self):
        """Get list of all available intents"""
        intents = [intent['tag'] for intent in self.intents_data['intents']]
        return intents

def main():
    """Test the chatbot engine"""
    print("Initializing chatbot...")
    chatbot = ChatbotEngine()
    
    print("\nAvailable intents:", chatbot.get_all_intents())
    print("\nType 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
        
        result = chatbot.chat(user_input)
        
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Bot: {result['response']}\n")

if __name__ == "__main__":
    main()
