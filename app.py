from flask import Flask, request, render_template, redirect 
import pickle
import pandas as pd 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Model ko load karna
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)
        
        # Assume CSV file hai
        data = pd.read_csv(file_path)
        
        # Drop unnecessary columns
        #data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        # Ensure karein ke sirf wohi features hon jo training ke waqt use hue the
        # Example: agar model ko fit karte waqt specific columns use hue the
        # for col in training_features:
        #     if col not in data.columns:
        #         data[col] = 0  # Ya kisi appropriate default value se fill karen
        # data = data[training_features]
        
        # Model se predict karna
        predictions = model.predict(data)
        
        # Convert predictions to list
        predictions = predictions.tolist()
        
        # Result ko HTML page par show karna
        return render_template('form.html', predictions=predictions)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
