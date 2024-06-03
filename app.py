from flask import Flask, request,flash, jsonify, abort
import os
import sys
import io
import shutil
from PIL import Image
from historical_weather_forecast_ankara import predict_weather
from cnn_prediction import predict_images
from flask_cors import CORS

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
CORS(app)

@app.route('/image/send', methods=['POST'])
def upload_image():
    files = request.files.getlist('file')  # Get a list of uploaded files

    # If there is no files
    if len(files) == 0:
        abort(400)

    #If files are not images
    for file in files:
        if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            abort(400)

    result_dict = {}

    input_folder_path = './input_folder'

    if os.path.exists(input_folder_path):
        shutil.rmtree(input_folder_path)

    os.makedirs(input_folder_path)

    # Iterate over each file and process it
    for file in files:
        # Read the file as bytes
        img_bytes = file.read()
        # Decode the numpy array as an image using PIL
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_pil.save(os.path.join(input_folder_path, file.filename))

    result_dict = predict_images()

    return jsonify(result_dict)

@app.route('/predict/weather', methods=['POST'])
def gaussian_predict():
    weather_conditions = request.get_json()
    response = []

    for parameter in weather_conditions:

        condition = predict_weather(
                           checkValidData(parameter.get("tempmax")),
                           checkValidData(parameter.get("tempmin")),
                           checkValidData(parameter.get("temp")),
                           checkValidData(parameter.get("feelslikemax")),
                           checkValidData(parameter.get("feelslikemin")),
                           checkValidData(parameter.get("feelslike")),
                           checkValidData(parameter.get("dew")),
                           checkValidData(parameter.get("humidity")),
                           checkValidData(parameter.get("precip")),
                           checkValidData(parameter.get("precipprob")),
                           checkValidData(parameter.get("precipcover")),
                           checkValidData(parameter.get("snow")),
                           checkValidData(parameter.get("snowdepth")),
                           checkValidData(parameter.get("windspeed")),
                           checkValidData(parameter.get("winddir")),
                           checkValidData(parameter.get("sealevelpressure")),
                           checkValidData(parameter.get("cloudcover")),
                           checkValidData(parameter.get("visibility")),
                           checkValidData(parameter.get("solarradiation")),
                           checkValidData(parameter.get("solarenergy")),
                           checkValidData(parameter.get("uvindex")),
                           checkValidData(parameter.get("moonphase"))
            )

        response.append(
            {
                'id': parameter.get("id"),
                'condition': condition
            }
        )
    return jsonify(response)

def checkValidData(data):
    if(type(data) == str):
        if(data == ''):
            return '-'
        else:
            return data
    else:
        return data


if __name__ == '__main__':
    app.run()
