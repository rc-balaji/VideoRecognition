import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from handsign import HandSignModel  # Assuming your model code is in models.py

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB limit
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}


CHECK_POINT = "./models/hs_model.ckpt"
LABELS = "./models/labels.txt"

# Initialize model
model = HandSignModel(
    checkpoint_path=CHECK_POINT,
    label_file=LABELS,
)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            result = model.predict_from_path(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up - remove the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
