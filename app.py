import os
import sys
import time
import uuid
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from communication_pipeline import process_itantra_pipeline, get_encodec_model, get_asr_model, AudioPipelineError

# Reconfigure stdout for UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
STATIC_AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_AUDIO_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        "status": "ONLINE",
        "system": "iTantra Neural Voice Communication Engine",
        "version": "1.0.0",
        "supported_languages": ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"],
        "supported_bitrates": [24.0, 12.0, 6.0, 3.0, 1.5]
    })

@app.route('/api/transmit', methods=['POST'])
def api_transmit():
    temp_path = None
    try:
        bitrate = float(request.form.get('bitrate', 6.0))
        language = request.form.get('language', 'hi')
        mode = request.form.get('mode', 'software_loopback')

        # Check if audio file was uploaded
        if 'audio' in request.files:
            file = request.files['audio']
            if not file.filename:
                return jsonify({
                    "status": "FAILURE",
                    "stage": "audio_upload",
                    "error": "Empty filename in uploaded audio payload"
                }), 400

            orig_ext = os.path.splitext(file.filename)[1].lower()
            if not orig_ext or len(orig_ext) > 10:
                orig_ext = '.bin'

            temp_filename = f"upload_{uuid.uuid4().hex}{orig_ext}"
            temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            file.save(temp_path)
        else:
            # Fallback to project audio.wav if no file provided
            temp_path = os.path.join(os.path.dirname(__file__), 'audio.wav')
            if not os.path.exists(temp_path):
                return jsonify({
                    "status": "FAILURE",
                    "stage": "audio_upload",
                    "error": "No audio file uploaded and default audio.wav not found"
                }), 400

        # Execute full iTantra pipeline
        result = process_itantra_pipeline(
            audio_path=temp_path,
            target_bitrate=bitrate,
            language=language,
            mode=mode,
            output_dir=STATIC_AUDIO_FOLDER
        )

        return jsonify(result)

    except AudioPipelineError as ape:
        print(f"[API Stage Error] Stage: {ape.stage}, Message: {ape.message}")
        return jsonify({
            "status": "FAILURE",
            "stage": ape.stage,
            "error": ape.message
        }), 400

    except Exception as e:
        print(f"[API Internal Error] {e}")
        return jsonify({
            "status": "FAILURE",
            "stage": "server_error",
            "error": "Internal processing error encountered during transmission"
        }), 500

    finally:
        # Clean up temporary uploaded file safely
        if temp_path and os.path.exists(temp_path) and temp_path.startswith(UPLOAD_FOLDER):
            try:
                os.remove(temp_path)
            except Exception as clean_err:
                print(f"[Cleanup Warning] {clean_err}")

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(STATIC_AUDIO_FOLDER, filename)

if __name__ == '__main__':
    get_encodec_model()
    print("[iTantra Server] Ready on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

