# STEP 1 — Importing all required libraries.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# STEP 2 — Loading the Dataset
# Load the CSV file into a pandas dataframe
df = pd.read_csv('indian_nonindian_names_200k.csv')

print("First 5 rows of dataset:")
print(df.head())

print("\nTotal names in dataset:", len(df))
print("\nLabel Distribution:")
print(df['label'].value_counts())

# STEP 3 — Preprocessing the Data
# Convert labels to numbers
# Indian     → 1
# Not Indian → 0
df['label'] = df['label'].map({'Indian': 1, 'Not Indian': 0})

print("Labels after conversion:")
print(df['label'].value_counts())

# Convert all names to lowercase
df['full_name'] = df['full_name'].str.lower()

print("\nSample names after lowercase:")
print(df['full_name'].head())

# Build character dictionary
# Every unique character gets a unique number
all_chars = set(''.join(df['full_name']))
char_to_index = {char: idx+1 for idx, char in enumerate(sorted(all_chars))}

print("\nTotal unique characters found:", len(char_to_index))
print("Sample character mapping:", dict(list(char_to_index.items())[:5]))

# Define max length
# All names will be padded to this length
MAX_LEN = 50

# Convert each name to a list of numbers
def name_to_numbers(name):
    return [char_to_index.get(char, 0) for char in name]

# Apply conversion to all names
X = df['full_name'].apply(name_to_numbers)
print("\nSample name converted to numbers:")
print(df['full_name'][0], "→", X[0])

# Pad all sequences to MAX_LEN
X = pad_sequences(X, maxlen=MAX_LEN, padding='post')
print("\nAfter padding shape:", X.shape)

# Get labels
y = df['label'].values
print("Labels shape:", y.shape)

# STEP 4 — Saving to the Dictionary and Spliting the Data
# Save the character dictionary using pickle
# We need this same dictionary in our API later!
with open('char_to_index.pkl', 'wb') as f:
    pickle.dump(char_to_index, f)

print("Character dictionary saved! ✅")

# Save vocab size for building the model
VOCAB_SIZE = len(char_to_index) + 1  # +1 for padding (0)
print("Vocabulary size:", VOCAB_SIZE)

# Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", len(X_train))
print("Testing data size :", len(X_test))


# STEP 5 — Building the LSTM Model
# Build the model layer by layer
model = Sequential()

# Layer 1 — Embedding Layer
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN))

# Layer 2 — LSTM Layer
model.add(LSTM(64, return_sequences=False))

# Layer 3 — Dropout Layer
model.add(Dropout(0.5))

# Layer 4 — Dense Layer
model.add(Dense(32, activation='relu'))

# Layer 5 — Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()


# STEP 6 — Train the Model
print("\nTraining started... Please wait! ⏳")
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\nTraining Complete! ✅")


# STEP 7 — Saving the Trained Model
# Save the trained model to a file
model.save('indian_name_model.keras')
print("Model saved successfully! ✅")

# Save MAX_LEN value for API use
with open('max_len.pkl', 'wb') as f:
    pickle.dump(MAX_LEN, f)
print("MAX_LEN saved successfully! ✅")

# Test the model with some example names
print("\n--- Testing Model with Examples ---")

def predict_name(name):
    # Convert to lowercase
    name = name.lower()
    # Convert to numbers
    name_numbers = [char_to_index.get(char, 0) for char in name]
    # Pad to MAX_LEN
    name_padded = pad_sequences([name_numbers], maxlen=MAX_LEN, padding='post')
    # Predict
    prediction = model.predict(name_padded, verbose=0)[0][0]
    # Return result
    if prediction > 0.5:
        return f"INDIAN NAME ✅ (Confidence: {prediction*100:.2f}%)"
    else:
        return f"NOT INDIAN NAME ❌ (Confidence: {(1-prediction)*100:.2f}%)"

# Test with Indian names
print(predict_name("Rajesh Kumar"))
print(predict_name("Priya Sharma"))
print(predict_name("Jyotisha Ranjan Dash"))

# Test with Non Indian names
print(predict_name("William Shakespeare"))
print(predict_name("John Smith"))
print(predict_name("Marie Curie"))