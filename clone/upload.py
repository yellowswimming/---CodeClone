from flask import Flask, request
from flask_cors import CORS
from multiprocess import result_mulsearch
from load_model import find_code_by_id
import time
import datetime

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/upload', methods=['POST'])
def upload():
    r = []
    start = time.time()
    file = request.files['file']
    file.save(file.filename)

    with open(file.filename, 'r') as f:
        content = f.read()
    ids = result_mulsearch(content)

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")
    r.append(date_string)
    r.append(str(int(time.time()-start))+"ms")
    r.append(find_code_by_id(list(ids.keys()),'code_data.csv'))
    r.append(ids)

    return r

@app.route('/code', methods=['POST'])
def code():
    r = []
    content = ""
    start = time.time()

    code = request.data.decode('utf-8')

    with open("helloWorld.cpp", 'w') as f:
        f.write(code)

    with open("helloWorld.cpp", 'r') as f:
        content = f.read()
    
    print(content)
    ids = result_mulsearch(content)

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")
    r.append(date_string)
    r.append(str(int(time.time()-start))+"ms")
    r.append(find_code_by_id(list(ids.keys()),'code_data.csv'))
    r.append(ids)

    return r

if __name__ == '__main__':
    app.run(debug=True)