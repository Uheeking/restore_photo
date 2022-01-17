from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

UPLOAD_DIR = "/Users/gidaehyeon/Downloads/PreviewContentTransition-main/venv/static"
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1', methods=['GET', 'POST'])
def index1():
    if request.method == 'POST':
        return render_template('index1.html')
    
@app.route('/index2', methods=['GET', 'POST'])
def index2():
    if request.method == 'POST':
        f = request.files['file']
        filnames = str(f.filename)
        filenames = str(f.filename.split('.')[-2])
        #path = os.path.join(app.config['UPLOAD_DIR'], filnames)
        f.save("/Users/gidaehyeon/"+secure_filename(f.filename))
        #f.save(secure_filename('/Users/gidaehyeon/Downloads/PreviewContentTransition-main/venv/static/img/' + secure_filename(filnames)))
        return render_template('index2.html',data1=f, data2=filnames, data3=filenames)

if __name__ ==  '__main__':
    app.run(host='0.0.0.0', port=8000)
