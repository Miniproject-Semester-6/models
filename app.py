from flask import Flask, request, jsonify
import src.models.forecast as forecast_model

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to the Forecasting Api!, It's open for all users."


@app.route("/forecast", methods=["POST"])
def forecast():
    if request.method == "POST":
        data = request.get_json()
        return jsonify(forecast_model.forecast(data, True))


if __name__ == "__main__":
    app.run(debug=True)
