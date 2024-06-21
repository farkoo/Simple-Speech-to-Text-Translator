from flask import Flask, request, jsonify
from hezar.models import Model
import os
import uuid
import time
from flasgger import Swagger
from pydub import AudioSegment

app = Flask(__name__)
swagger = Swagger(app)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = '/root/farkoo/tmp'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = Model.load("hezarai/whisper-small-fa")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_ogg_to_mp3(ogg_file_path, mp3_file_path):
    """
    Converts an OGG file to MP3 format.

    :param ogg_file_path: Path to the input OGG file.
    :param mp3_file_path: Path to the output MP3 file.
    """
    try:
        # Load the OGG file
        audio = AudioSegment.from_file(ogg_file_path, format="ogg")

        # Export as MP3
        audio.export(mp3_file_path, format="mp3")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file to text
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The audio file to upload
    responses:
      200:
        description: The transcribed text
        schema:
          type: object
          properties:
            text:
              type: string
      400:
        description: Invalid input
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}_{int(time.time())}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        mp3_filepath = None
        try:
            if file_ext == 'ogg':
                mp3_filename = f"{uuid.uuid4()}_{int(time.time())}.mp3"
                mp3_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)
                if convert_ogg_to_mp3(filepath, mp3_filepath):
                    final_filepath = mp3_filepath
                else:
                    os.remove(filepath)
                    return jsonify({'error': 'Failed to convert OGG to MP3'}), 500
            else:
                final_filepath = filepath

            transcripts = model.predict(final_filepath)
            text = transcripts[0]['text'] if 'text' in transcripts[0] else ""

        finally:
            # Delete the processed files
            if os.path.exists(filepath):
                os.remove(filepath)
            if mp3_filepath and os.path.exists(mp3_filepath):
                os.remove(mp3_filepath)

        response = jsonify({'text': text})
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
