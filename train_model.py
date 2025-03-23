import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data.csv")

# Split dataset
X = df[['Size (sq ft)', 'Bedrooms', 'Age (years)']]
y = df['Price ($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'model.pkl'")
