from flask import Flask, render_template, request, Response, send_file
import noise


upload_file = "/Users/gidaehyeon/Projects/venv/static/img/"
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1', methods=['GET', 'POST'])
def index1():
    global main_1
    if request.method == 'POST':
        main_1 = request.form['main']
        return render_template('index1.html', data=main_1)
    
@app.route('/index2', methods=['GET', 'POST'])
def index2():
    global main_1
    if request.method == 'POST':
        f = request.files['file']
        filnames = str(f.filename)
        filenames = str(f.filename.split('.')[-2])
        f.save(upload_file+secure_filename(f.filename))
        
        name = request.form['text']
        if request.form['secret'] == "Image_Denoising":
            noise.main(upload_file+secure_filename(f.filename), name)
        elif request.form['secret'] == "Convolution_blurring":
            noise.main2(upload_file+secure_filename(f.filename), name)
        elif request.form['secret'] == "Gaussian_Denoising":
            noise.main3(upload_file+secure_filename(f.filename), name)
        elif request.form['secret'] == "Bilateral_filter":
            noise.main4(upload_file+secure_filename(f.filename), name)
        else:
            noise.main5(upload_file+secure_filename(f.filename), name)
        return render_template('index2.html', name=name)
    
@app.route('/index_home', methods=['GET', 'POST'])
def index_home():
    if request.method == 'POST':
        return render_template('index_home.html')

if __name__ ==  '__main__':
    app.run(host='0.0.0.0', port=8000)
