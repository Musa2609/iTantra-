import os
import sys
import time
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from communication_pipeline import process_itantra_pipeline, get_encodec_model, get_asr_model

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
    try:
        bitrate = float(request.form.get('bitrate', 6.0))
        language = request.form.get('language', 'hi')
        mode = request.form.get('mode', 'software_loopback')

        # Check if audio file was uploaded
        if 'audio' in request.files:
            file = request.files['audio']
            temp_filename = f"upload_{int(time.time()*1000)}.wav"
            temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            file.save(temp_path)
        else:
            # Fallback to project audio.wav if no file provided
            temp_path = os.path.join(os.path.dirname(__file__), 'audio.wav')
            if not os.path.exists(temp_path):
                return jsonify({"error": "No audio file uploaded and default audio.wav not found"}), 400

        # Execute full iTantra pipeline
        result = process_itantra_pipeline(
            audio_path=temp_path,
            target_bitrate=bitrate,
            language=language,
            mode=mode,
            output_dir=STATIC_AUDIO_FOLDER
        )

        return jsonify(result)

    except Exception as e:
        print(f"[API Error] {e}")
        return jsonify({"error": str(e), "status": "FAILURE"}), 500

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(STATIC_AUDIO_FOLDER, filename)

if __name__ == '__main__':
    # Pre-warm models on server launch
    get_encodec_model()
    print("[iTantra Server] Ready on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
