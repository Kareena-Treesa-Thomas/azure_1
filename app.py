
import os
import base64
from flask import Flask, request, jsonify, send_from_directory
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

VISION_KEY = os.environ.get("VISION_KEY", "")
VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "").rstrip("/")

# Face API can be a separate resource/key. Falls back to VISION_KEY/ENDPOINT
# if you provisioned Face as part of a multi-service resource.
FACE_KEY = os.environ.get("FACE_KEY", VISION_KEY)
FACE_ENDPOINT = os.environ.get("FACE_ENDPOINT", VISION_ENDPOINT).rstrip("/")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


def get_image_bytes(image_url, image_base64):
    """Return (bytes_or_none, url_or_none) so every downstream call can
    work the same way regardless of whether the user uploaded a file
    or pasted a URL."""
    if image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        return base64.b64decode(image_base64), None
    if image_url:
        return None, image_url
    return None, None


def call_v32_analyze(image_bytes, image_url, features):
    """features e.g. 'Brands' or 'Categories'. details=Landmarks always added."""
    v32_url = (
        f"{VISION_ENDPOINT}/vision/v3.2/analyze"
        f"?visualFeatures={features}&details=Landmarks"
    )
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    if image_url:
        headers["Content-Type"] = "application/json"
        resp = requests.post(v32_url, headers=headers, json={"url": image_url}, timeout=20)
    else:
        headers["Content-Type"] = "application/octet-stream"
        resp = requests.post(v32_url, headers=headers, data=image_bytes, timeout=20)
    resp.raise_for_status()
    return resp.json()


def call_face_detect(image_bytes, image_url):
    face_url = (
        f"{FACE_ENDPOINT}/face/v1.0/detect"
        f"?returnFaceId=true&returnFaceLandmarks=false"
        f"&returnFaceAttributes=age,gender,emotion"
        f"&detectionModel=detection_03"
    )
    headers = {"Ocp-Apim-Subscription-Key": FACE_KEY}
    if image_url:
        headers["Content-Type"] = "application/json"
        resp = requests.post(face_url, headers=headers, json={"url": image_url}, timeout=20)
    else:
        headers["Content-Type"] = "application/octet-stream"
        resp = requests.post(face_url, headers=headers, data=image_bytes, timeout=20)
    resp.raise_for_status()
    return resp.json()


@app.route("/analyze", methods=["POST"])
def analyze():
    if not VISION_KEY or not VISION_ENDPOINT:
        return jsonify({
            "error": "Azure Vision not configured. Set VISION_KEY and "
                     "VISION_ENDPOINT in App Service -> Configuration -> "
                     "Application settings, then restart the app."
        }), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")

    image_bytes, resolved_url = get_image_bytes(image_url, image_base64)
    if image_bytes is None and resolved_url is None:
        return jsonify({"error": "No image URL or image data provided"}), 400

    # --- Base call: tags, OCR, caption (v4.0 Image Analysis) ---
    api_url = (
        f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
        f"?api-version=2023-10-01&features=Tags,Read"
    )
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    try:
        if resolved_url:
            headers["Content-Type"] = "application/json"
            resp = requests.post(api_url, headers=headers, json={"url": resolved_url}, timeout=20)
        else:
            headers["Content-Type"] = "application/octet-stream"
            resp = requests.post(api_url, headers=headers, data=image_bytes, timeout=20)
    except requests.RequestException as exc:
        return jsonify({"error": f"Request to Azure AI Vision failed: {exc}"}), 502

    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text or "Unexpected response from Azure AI Vision"}

    # --- Landmark detection (v3.2, nested in categories) ---
    try:
        landmark_data = call_v32_analyze(image_bytes, resolved_url, "Categories")
        landmarks = []
        for cat in landmark_data.get("categories", []):
            detail = cat.get("detail", {})
            for lm in detail.get("landmarks", []):
                landmarks.append(lm)  # {"name": ..., "confidence": ...}
        body["landmarks"] = landmarks
    except requests.RequestException as exc:
        body["landmarks"] = []
        body["landmarks_error"] = str(exc)

    # --- Brand detection (v3.2) ---
    try:
        brand_data = call_v32_analyze(image_bytes, resolved_url, "Brands")
        body["brands"] = brand_data.get("brands", [])
    except requests.RequestException as exc:
        body["brands"] = []
        body["brands_error"] = str(exc)

    # --- Face detection (separate Face API) ---
    try:
        faces = call_face_detect(image_bytes, resolved_url)
        body["faces"] = faces
    except requests.RequestException as exc:
        body["faces"] = []
        body["faces_error"] = str(exc)

    return jsonify(body), resp.status_code


@app.route("/health")
def health():
    return jsonify({"status": "ok", "configured": bool(VISION_KEY and VISION_ENDPOINT)})


if __name__ == "__main__":
    app.run(debug=True)