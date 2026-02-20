from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model you just moved into this folder
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    # Shows the user the Home page (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the engagement metrics entered by the user
    features = [float(x) for x in request.form.values()]
    
    # 2. Analyze the inputs using the Random Forest model
    prediction = model.predict([np.array(features)])
    
    # 3. Determine the result (1 = Successful, 0 = Likely to Fail)
    output = "Successful" if prediction[0] == 1 else "Likely to Fail"
    
    # 4. Show the prediction on the results page
    return render_template('index.html', prediction_text=f'Startup Status: {output}')

if __name__ == "__main__":
    app.run(debug=True)