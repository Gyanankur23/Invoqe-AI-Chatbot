"""
Generate analysis plots for Task 2 - Chatbot
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('data/chatbot_data.csv')

# Create plots directory
os.makedirs('plots', exist_ok=True)

# 1. Intent distribution
plt.figure(figsize=(10, 6))
intent_counts = df['intent'].value_counts()
intent_counts.plot(kind='bar')
plt.title('Intent Distribution')
plt.xlabel('Intent')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/intent_distribution.png')
plt.close()

# 2. Pattern length distribution
df['pattern_length'] = df['pattern'].str.len()
plt.figure(figsize=(10, 6))
plt.hist(df['pattern_length'], bins=20, edgecolor='black')
plt.title('Pattern Length Distribution')
plt.xlabel('Pattern Length (characters)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/pattern_length_distribution.png')
plt.close()

print("Plots generated in plots/ directory")
