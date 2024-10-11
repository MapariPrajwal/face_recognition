# app.py

from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('face_recognition/fr_2.py')
def index():
    return render_template('index.html')

@app.route('/runcode', methods=['POST'])
def run_code():
    subprocess.run(['python', 'face_recognition/fr_2.py'])
    return 'Python script executed successfully!'

if __name__ == '__main__':
    app.run(debug=True)
