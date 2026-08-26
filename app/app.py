import os
import sys
import webbrowser
import threading
import joblib
from flask import Flask, render_template, request, jsonify

# Add project root directory to path for unpickling custom transformer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.feature_engineering import preprocess_single_input, TitanicFeatureEngineer

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load Trained Model Pipeline
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'titanic_pipeline.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model pipeline not found at {MODEL_PATH}. Please run 'python src/train.py' first.")

model_pipeline = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form.get('Pclass', 3))
        sex = request.form.get('Sex', 'male')
        age = float(request.form.get('Age', 28.0))
        sibsp = int(request.form.get('SibSp', 0))
        parch = int(request.form.get('Parch', 0))
        fare = float(request.form.get('Fare', 14.45))
        embarked = request.form.get('Embarked', 'S')

        input_df = preprocess_single_input(
            pclass=pclass,
            sex=sex,
            age=age,
            sibsp=sibsp,
            parch=parch,
            fare=fare,
            embarked=embarked
        )

        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]
        survival_prob = round(probabilities[1] * 100, 1)

        result_data = {
            'status': 'Survived ✅' if prediction == 1 else 'Did Not Survive ❌',
            'is_survived': bool(prediction == 1),
            'probability': survival_prob,
            'pclass': pclass,
            'sex': sex.capitalize(),
            'age': age,
            'fare': fare,
            'embarked': embarked
        }

        return render_template('index.html', result=result_data)
    except Exception as e:
        error_msg = f"Error processing prediction: {str(e)}"
        return render_template('index.html', result={'error': error_msg})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        input_df = preprocess_single_input(
            pclass=data.get('Pclass', 3),
            sex=data.get('Sex', 'male'),
            age=data.get('Age', 28.0),
            sibsp=data.get('SibSp', 0),
            parch=data.get('Parch', 0),
            fare=data.get('Fare', 14.45),
            embarked=data.get('Embarked', 'S')
        )
        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'status': 'Survived' if prediction == 1 else 'Did Not Survive',
            'survival_probability_percentage': round(probabilities[1] * 100, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Auto open browser after 1.2 seconds when started directly
    threading.Timer(1.2, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
