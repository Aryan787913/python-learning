import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# ==============================
# Create Dataset
# ==============================

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy',
                'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy',
                'Sunny', 'Overcast', 'Overcast', 'Rainy'],

    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],

    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],

    'Windy': ['False', 'True', 'False', 'False', 'False',
              'True', 'True', 'False', 'False', 'False',
              'True', 'True', 'False', 'True'],

    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes',
                   'No', 'Yes', 'No', 'Yes', 'Yes',
                   'Yes', 'Yes', 'Yes', 'No']
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

print("\nOriginal Dataset:\n")
print(df)

# ==============================
# Label Encoding
# ==============================

encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

print("\nEncoded Dataset:\n")
print(df)

# ==============================
# Features and Target
# ==============================

X = df.drop('PlayTennis', axis=1)
Y = df['PlayTennis']

# ==============================
# Create and Train Model
# ==============================

model = CategoricalNB()
model.fit(X, Y)

# ==============================
# Training Accuracy
# ==============================

y_pred = model.predict(X)

accuracy = accuracy_score(Y, y_pred)

print("\nTraining Accuracy: {:.2f}%".format(accuracy * 100))

# ==============================
# New Prediction
# Sunny, Cool, High, True
# ==============================

sample_df = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Windy': ['True']
})

# Encode sample data
for column in sample_df.columns:
    sample_df[column] = encoders[column].transform(sample_df[column])

# Predict
prediction = model.predict(sample_df)

# Convert prediction back to Yes/No
result = encoders['PlayTennis'].inverse_transform(prediction)

print("\nPrediction for (Sunny, Cool, High, True):", result[0])
