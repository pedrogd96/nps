from flask import Flask, request, jsonify
from src.utils.config_loader import load_config
from src.model.predict import predict

app = Flask(__name__)
config = load_config("configs/api.yaml")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        request_data = request.get_json()
        comment = request_data.get("comentario")

        if comment is None:
            return jsonify({"error": "Campo 'comentario' é obrigatório"}), 400

        result = predict(comment);

        return jsonify({
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(
        host=config["api"]["host"],
        port=config["api"]["port"],
        debug=True
    )