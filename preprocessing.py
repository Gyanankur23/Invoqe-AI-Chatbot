"""
Simple Text Preprocessing for Chatbot
"""

import pandas as pd
import re
import os

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(input_file='data/chatbot_data.csv', output_file='data/processed_data.csv'):
    """Preprocess the entire dataset"""
    print("=" * 50)
    print("TEXT PREPROCESSING")
    print("=" * 50)
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples")
    
    df['processed_pattern'] = df['pattern'].apply(clean_text)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    print(f"\nOriginal: {df['pattern'].iloc[0]}")
    print(f"Processed: {df['processed_pattern'].iloc[0]}")
    
    return df

def main():
    preprocess_dataset()

if __name__ == "__main__":
    main()
