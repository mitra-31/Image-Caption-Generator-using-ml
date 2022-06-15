from flask import Flask,render_template,redirect,url_for,flash,request
import urllib.request
import os
from predictions.model import *


app = Flask(__name__,template_folder="template")
UPLOAD_FOLDER = 'static/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("base.html")


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        name = "test."
        filename = name + "jpeg"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        result = run_model()
        return render_template('base.html', display="yes",result=result)
    else:
        return redirect(request.url)

 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename), code=301)
 

if __name__ == "__main__":
    app.run(port=8080)