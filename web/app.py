from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import joblib

app = Flask(__name__)
api = Api(app)

model = joblib.load("model.pkl")    

class Predict(Resource):
    def post(self):

        postedData = request.get_json()
        temperature = int(postedData["temperature"])

        y_pred = model.predict([[temperature]])

        if y_pred > 10:
            fanSpeed = 10
        elif y_pred < 0:
            fanSpeed = 0
        else:
            fanSpeed = int(y_pred[0][0])    

        retJson = {
        "status": 200,
        "msg": "success",
        "fanSpeed" : fanSpeed
        }

        return jsonify(retJson)

api.add_resource(Predict, "/predict")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)