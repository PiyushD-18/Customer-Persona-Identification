import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Create the GUI window
window = tk.Tk()
window.title("Customer Persona Prediction")
window.configure(bg="black")

# Dummy data for demonstration
data = pd.DataFrame({
    'Age': [25, 35, 45, 30, 28, 42, 55, 38, 48, 50],
    'Income': [50000, 70000, 90000, 60000, 55000, 85000, 95000, 80000, 92000, 100000],
    'Purchase_History': ['Low', 'Medium', 'High', 'Low', 'Low', 'High', 'High', 'Medium', 'Medium', 'High'],
    'Interests': ['Sports', 'Travel', 'Fashion', 'Sports', 'Food', 'Fashion', 'Travel', 'Sports', 'Travel', 'Food'],
    'Persona': ['Young Professionals', 'Young Professionals', 'Middle-Aged Professionals',
                'Young Professionals', 'Young Professionals', 'Middle-Aged Professionals',
                'Senior Professionals', 'Middle-Aged Professionals', 'Middle-Aged Professionals',
                'Senior Professionals']
})

# Feature selection
features = ['Age', 'Income', 'Purchase_History', 'Interests']
target = 'Persona'

# Encode categorical variables
data_encoded = pd.get_dummies(data[features])

# Random Forest Model Training
X = data_encoded
y = data[target]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Function to predict persona
def predict_persona():
    age = age_var.get()
    income = income_var.get()
    purchase_history = purchase_history_combobox.get()
    interests = interests_combobox.get()

    new_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Purchase_History': [purchase_history],
        'Interests': [interests]
    })

    # Encode new data
    new_data_encoded = pd.get_dummies(new_data)
    new_data_encoded = new_data_encoded.reindex(columns=data_encoded.columns, fill_value=0)

    # Predict persona
    predicted_persona = rf_model.predict(new_data_encoded)

    result_label.configure(text="Predicted Persona: " + predicted_persona[0])

# Create input labels and entry fields
age_label = tk.Label(window, text="Age:", font=("Arial", 12), fg="white", bg="black")
age_label.pack(pady=10)
age_var = tk.IntVar()
age_entry = tk.Entry(window, textvariable=age_var, font=("Arial", 12), width=30)
age_entry.pack()

income_label = tk.Label(window, text="Income:", font=("Arial", 12), fg="white", bg="black")
income_label.pack(pady=10)
income_var = tk.IntVar()
income_entry = tk.Entry(window, textvariable=income_var, font=("Arial", 12), width=30)
income_entry.pack()

purchase_history_label = tk.Label(window, text="Purchase History:", font=("Arial", 12), fg="white", bg="black")
purchase_history_label.pack(pady=10)
purchase_history_var = tk.StringVar()
purchase_history_combobox = ttk.Combobox(window, textvariable=purchase_history_var, font=("Arial", 12), values=["Low", "Medium", "High"], width=27, state="readonly")
purchase_history_combobox.pack()

interests_label = tk.Label(window, text="Interests:", font=("Helvetica", 12), fg="white", bg="black")
interests_label.pack(pady=10)
interests_var = tk.StringVar()
interests_combobox = ttk.Combobox(window, textvariable=interests_var, font=("Arial", 12), values=["Sports", "Travel", "Fashion", "Food"], width=27, state="readonly")
interests_combobox.pack()

# Create predict button
predict_button = tk.Button(window, text="Predict Persona", command=predict_persona, font=("Arial", 14, "bold"), bg="blue", fg="white")
predict_button.pack(pady=20)

# Create result label
result_label = tk.Label(window, text="Predicted Persona: ", font=("Arial", 14, "bold"), fg="white", bg="black")
result_label.pack()

# Run the GUI main loop
window.mainloop()
