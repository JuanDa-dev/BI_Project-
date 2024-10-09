from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Cargar los modelos previamente entrenados
pipeline_nb = joblib.load('./pipeline_nb.pkl')
pipeline_svm = joblib.load('./pipeline_svm.pkl')
pipeline_rf = joblib.load('./pipeline_rf.pkl')

# Crear diccionario para almacenar el accuracy de cada modelo después del reentrenamiento
accuracy_scores = {
    'naive_bayes': None,
    'svm': None,
    'random_forest': None
}

app = Flask(__name__)

# Ruta para servir la página principal (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    textos = data['text']  # Lista de textos a clasificar
    model_name = data['model']  # Nombre del modelo ('naive_bayes', 'svm', 'random_forest')

    # Seleccionar el modelo correspondiente
    if model_name == 'naive_bayes':
        model = pipeline_nb
        accuracy = accuracy_scores['naive_bayes']
    elif model_name == 'svm':
        model = pipeline_svm
        accuracy = accuracy_scores['svm']
    elif model_name == 'random_forest':
        model = pipeline_rf
        accuracy = accuracy_scores['random_forest']
    else:
        return jsonify({'error': 'Modelo no reconocido'}), 400

    # Hacer la predicción
    predicciones = model.predict(textos)

    # Si el modelo tiene 'predict_proba', calcular y devolver las probabilidades
    if hasattr(model, 'predict_proba'):
        probabilidades = model.predict_proba(textos).tolist()
        probabilidades_format = [f'{p * 100:.2f}%' for p in probabilidades[0]]  # Convertir a porcentaje
    else:
        probabilidades_format = 'No disponible'

    # Devolver el accuracy y las predicciones
    return jsonify({
        'predicciones': predicciones.tolist(),
        'probabilidades': probabilidades_format,
        'accuracy': f'{accuracy * 100:.2f}%' if accuracy else 'Accuracy no disponible'
    }), 200
    
# Endpoint para re-entrenar el modelo
@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.get_json()
    textos = data['text']  # Características (X)
    etiquetas = data['label']  # Etiquetas (y)
    model_name = data['model']  # Nombre del modelo a reentrenar

    # Seleccionar el modelo correspondiente
    if model_name == 'naive_bayes':
        model = MultinomialNB()
        pipeline = pipeline_nb
    elif model_name == 'svm':
        model = SVC(kernel='linear', probability=True)  # Asegurarse de que SVM tiene probabilidades habilitadas
        pipeline = pipeline_svm
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        pipeline = pipeline_rf
    else:
        return jsonify({'error': 'Modelo no reconocido'}), 400

    # Re-entrenar el modelo
    X_train, X_test, y_train, y_test = train_test_split(textos, etiquetas, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # Guardar el nuevo modelo
    joblib.dump(pipeline, f'pipeline_{model_name}.pkl')

    # Hacer predicciones en el conjunto de prueba y calcular las métricas
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Guardar el accuracy en el diccionario
    accuracy_scores[model_name] = accuracy

    # Devolver las métricas
    return jsonify({
        'accuracy': f'{accuracy * 100:.2f}%',
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }), 200

if __name__ == '__main__':
    app.run(debug=True)