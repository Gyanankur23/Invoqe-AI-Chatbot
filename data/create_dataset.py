"""
Create training dataset for the AI Chatbot
Customer service domain with intents and responses
"""

import pandas as pd
import os

def create_chatbot_dataset():
    """Create a CSV dataset with patterns and intents for training"""
    
    data = []
    
    intents_data = [
        ("greeting", ["Hi", "Hello", "Hey", "Good morning", "Good afternoon", "Good evening", "Hi there", "Hello there"]),
        ("goodbye", ["Bye", "Goodbye", "See you later", "Have a good day", "Take care", "See ya", "I'm leaving"]),
        ("thanks", ["Thank you", "Thanks", "Thanks a lot", "Thank you very much", "I appreciate it", "Thanks for your help"]),
        ("product_info", ["What products do you offer?", "Tell me about your products", "What do you sell?", "Product information", "Show me your products", "What items are available?"]),
        ("pricing", ["How much does it cost?", "What is the price?", "Pricing information", "How much?", "Cost details", "Price of products"]),
        ("shipping", ["How long does shipping take?", "Shipping information", "Delivery time", "When will I receive my order?", "Shipping details", "How fast is delivery?"]),
        ("return_policy", ["What is your return policy?", "Can I return items?", "Return information", "How do I return something?", "Refund policy", "Return process"]),
        ("payment_methods", ["What payment methods do you accept?", "Payment options", "How can I pay?", "Accepted payment methods", "Credit card", "Payment types"]),
        ("support", ["I need help", "Customer support", "Contact support", "Help me", "I have a problem", "Technical support"]),
        ("order_status", ["Where is my order?", "Order status", "Track my order", "Check order", "Order delivery status", "When will my order arrive?"]),
        ("hours", ["What are your hours?", "When are you open?", "Business hours", "Opening times", "Store hours", "Operating hours"]),
        ("location", ["Where are you located?", "Your address", "Store location", "Find your store", "Where is your shop?", "Address"]),
        ("warranty", ["What is the warranty?", "Warranty information", "Product warranty", "How long is the warranty?", "Warranty coverage", "Guarantee"]),
        ("discount", ["Do you have any discounts?", "Any promotions?", "Discount codes", "Special offers", "Sales", "Deals"]),
        ("default", ["I don't understand", "What do you mean?", "Can you clarify?", "Not sure", "Confused"])
    ]
    
    for intent, patterns in intents_data:
        for pattern in patterns:
            data.append({"pattern": pattern, "intent": intent})
    
    df = pd.DataFrame(data)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/chatbot_data.csv', index=False)
    
    print(f"Dataset created and saved to data/chatbot_data.csv")
    print(f"Total samples: {len(df)}")
    print(f"Total intents: {df['intent'].nunique()}")
    
    return df

if __name__ == "__main__":
    create_chatbot_dataset()
