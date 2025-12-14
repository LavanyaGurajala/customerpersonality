from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict_cluster import predict_customer_segment

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_demo_purposes' # Required for session

@app.route('/')
def home():
    # Retrieve result from session and clear it (pop)
    # This ensures that refreshing the page removes the result
    prediction = session.pop('prediction', None)
    error = session.pop('error_message', None)
    return render_template('index.html', prediction=prediction, error=error)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form.get('age'))
        income = request.form.get('income')
        # Handle optional income: convert to float only if string is not empty
        income = float(income) if income and income.strip() else None
        
        marital_status = request.form.get('marital_status')
        education = request.form.get('education')
        total_spent = float(request.form.get('total_spent'))
        num_purchases = int(request.form.get('num_purchases'))
        
        # Get prediction
        result = predict_customer_segment(
            age=age,
            income=income,
            marital_status=marital_status,
            education=education,
            total_spent=total_spent,
            num_purchases=num_purchases
        )
        
        # Store result in session
        session['prediction'] = result
        return redirect(url_for('home'))
    
    except Exception as e:
        session['error_message'] = f"Error processing your request: {str(e)}"
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
