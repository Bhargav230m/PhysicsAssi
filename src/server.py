from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyttsx3
from physicsAssi import generate_response
import os

print("Loading Models..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
engine = pyttsx3.init()
model_2 = whisper.load_model("small")
model.to(device)
model_2.to(device)

# History Variable, Current version only supports a single history and chat at once
current_count = 0
history = {}
assistant_count = 1

app = Flask("PhysicsAssi")
CORS(app)

@app.route('/get_voice', methods=['POST'])
def get_voice():
    if request.method == 'POST':
        audio_data = request.data
        saved_audio_file = 'saved_voice.wav'
        response_audio_file = 'response.wav'
        try:
            with open(saved_audio_file, 'wb') as f:
                f.write(audio_data)
            content = generate_response(model, tokenizer, model_2, engine, device, saved_audio_file, history, current_count, assistant_count)
            if os.path.exists(response_audio_file):
                return jsonify({ "message": "Audio data received and processed.", "content": content}), 200
            else:
                return "Processing failed.", 500
        except Exception as e:
            print("Failed to save file:", str(e))
            return "Failed to save file.", 500

@app.route('/get_response_audio', methods=['GET'])
def get_response_audio():
    response_audio_file = 'response.wav'
    if os.path.exists(response_audio_file):
        return send_file(response_audio_file, mimetype='audio/wav')
    else:
        return "Response audio not found.", 404
    
@app.route('/clear_history', methods=['POST'])
def clear_history():
    global current_count, history, assistant_count

    current_count = 0
    assistant_count = 1
    history = {}

    return "Successfully cleared history"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
