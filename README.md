# Audio Transcription Service with Flask

This repository contains a simple Flask web application that provides an endpoint for transcribing audio files using a pre-trained model.

## Overview

The application allows users to upload audio files in various formats, including WAV, MP3, M4A, and FLAC. Once an audio file is uploaded, the application uses the hezarai/whisper-small-fa model to transcribe the audio content and returns the transcripts in JSON format.


Installation

1.    Clone this repository to your local machine.
2.    Ensure you have Python installed (version 3.6 or later).
3.    Install the required Python packages:

```

pip install Flask hezar

```

## Usage

1.    Run the Flask application:

```

python main.py

```

2. The application will start a web server accessible at http://127.0.0.1:5000.

3. Example Request:
```
   curl -X POST http://127.0.0.1:5000/transcribe -F "file=@path_to_audio_file.wav"
```

4. Example Response:
```
{
    "transcripts": [
        {
            "chunks": null,
            "text": "و این تنها محدود به محیط کار نیست"
        }
    ]
}
```

## Support

**Contact me @:**

e-mail:

* farzanehkoohestani2000@gmail.com

Telegram id:

* [@farzaneh_koohestani](https://t.me/farzaneh_koohestani)
