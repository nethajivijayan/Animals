from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from model import predict
app = Flask(__name__)
CORS(app, origins="*")


@app.route('/')
def hello_world():
    return 'Hello World'


@app.post("/validate")
def validate():
    file = request.files.get('file')
    periodicity = request.form.get('periodicity').lower()
    target = request.form.get('target').lower()
    file.save(file.name)
    res = predict(file.name, periodicity, target)
    return jsonify({"image": f"http://localhost:5000{url_for('static', filename=res['output'])}", "data": f"http://localhost:5000{url_for('static', filename=res['data'])}", "analyze": res['analyze']})


if __name__ == '__main__':
    app.run(debug=True)