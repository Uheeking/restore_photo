from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1', methods=['GET', 'POST'])
def index1():
    if request.method == 'POST':
        f = request.files['file']
        names = request.form['naming']
        filnames = str(f.filename)
        return render_template('index1.html', data1=f)

if __name__ ==  '__main__':
    app.run()