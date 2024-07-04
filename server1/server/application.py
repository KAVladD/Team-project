from flask import Flask
from flask import render_template
from flask import request
from flask import abort
import datetime
from flask import jsonify
import matplotlib.pyplot as plt
import io
from PIL import Image
import joblib
import numpy as np
app = Flask(__name__)
dataList = []
model = joblib.load('model_weights.pkl')

@app.route('/')
def index():
    return render_template("main.html")

def predict_park(model, mean , std, freq):
    predict = model.predict(np.array([float(mean), float(std), freq]).reshape(1, -1) )
    if predict[0] == 0:
        res = 'Здоров'
    else:
        res = 'Болен'
        print(predict[0])
    return res

@app.route('/data')
def data_table():
    app.logger.info(dataList)
    app.logger.info("*********************")
    app.logger.info(type(dataList))
    app.logger.info("*********************")
    orderedDatList = dataList.copy()
    orderedDatList.reverse()
    return render_template("dataTemplate.html", data=orderedDatList)


@app.route('/data/create', methods=['POST'])
def create_task():
    if not request.json:
        abort(400)
    pred = predict_park(model, request.json['mean'], request.json['std'], request.json['freq'] )
    data = {
        'numbers': str(request.json['values']),
        'mean':str(request.json['mean']),
        'std':str(request.json['std']),
        'freq':str(request.json['freq']),
        'time': str(datetime.datetime.now().replace(microsecond=0)).replace(":", "_"),
        'plot': "none",
        'pred': pred
    }
    create_graphs(data)
    dataList.append(data)
    app.logger.info(data)
    return jsonify({"success": True})


def parse_string_to_list(input_string):
    input_string = input_string.replace("[", "").replace("]", "")
    input_list = input_string.split(",")
    result_list = [float(num.strip()) for num in input_list]
    return result_list


def create_graphs(data):
    data["plot"] = "./static/plots/" + data["time"].replace(" ", "_") + ".png"
    floatArray = parse_string_to_list(data["numbers"])
    time = np.arange(len(floatArray))/30
    plt.plot(time, floatArray)
    fig = plt.gcf()
    img = fig2img(fig)
    img.save(data["plot"])
    plt.close(fig)


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
