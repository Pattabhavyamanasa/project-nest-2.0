from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocess the data
encoder = LabelEncoder()
data["risk_level"] = encoder.fit_transform(data["risk level"])
data["disease"] = encoder.fit_transform(data["disease"])
data["symptoms"] = encoder.fit_transform(data["symptoms"])
data["cures"] = encoder.fit_transform(data["cures"])
data["doctor"] = encoder.fit_transform(data["doctor"])

# Split data into features and target
x = data[['disease', 'cures']]
y = data["symptoms"]

# Train the model
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
try:
    tree_clf.fit(x, y)
except ValueError as e:
    # Handle unseen label error
    @app.route('/')
    def error_handler():
        return jsonify({'error': str(e)})

    if __name__ == '__main__':
        app.run(debug=True)
    exit()

# Save the trained model using pickle
with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(tree_clf, model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the disease entered by the user from the form
    disease = request.form['disease']
    
    # Convert the disease to the encoded value
    disease_encoded = encoder.transform([disease])[0]
    
    # Perform prediction
    try:
        symptom_prediction = tree_clf.predict([[disease_encoded, 0]])  # Assuming symptoms are not considered
        # Convert predicted symptom back to original label
        predicted_symptom = encoder.inverse_transform(symptom_prediction)[0]
    except ValueError as e:
        # Handle unseen label error
        return jsonify({'error': str(e)})
    
    # You need to define how to get risk level from prediction
    risk_level = ["low(0.1%,0.5%)","moderate(1%)","high(20%,15%,70%)"]

    # You need to define how to get the actual symptom names from prediction
    disease_name = ["flu","bronchitis","pneumonia","heart attack","stroke","cancer","diabetes","arthritis","adenovirus","anemia","anxiety disorder","asthma","appendictis","bipolar disorder","blood clot","bursitis","cervical cancer","chickenpox","cholestrol","depression","diarrhea","food poisining"]

    return jsonify({
        'disease': disease_name[predicted_symptom],
        'risk_level': risk_level
    })

if __name__ == '__main__':
    app.run(debug=True) 