import os
import io
import json
import re
import yaml
import requests
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from pydantic import Field
from typing import List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Translation imports
from transformers import MarianMTModel, MarianTokenizer

# TTS imports
import pyttsx3
import tempfile
from gtts import gTTS

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

API_KEY        = cfg["api_key"]
BASE_URL       = cfg["model_server_base_url"].rstrip("/")
WORKSPACE_SLUG = cfg["workspace_slug"]
MODEL_NAME     = cfg["model_name"]
STREAM         = cfg.get("stream", False)
STREAM_TIMEOUT = cfg.get("stream_timeout", 60)

app = Flask(__name__)
CORS(app)

# Translation models setup
MODELS = {
    'en-hi': {
        'model_name': 'Helsinki-NLP/opus-mt-en-hi',
        'tokenizer': None,
        'model': None
    },
    'en-ta': {
        'model_name': 'Helsinki-NLP/opus-mt-en-ta',
        'tokenizer': None,
        'model': None
    }
}

def load_model(lang_pair):
    """Load translation model for specific language pair"""
    if lang_pair in MODELS and MODELS[lang_pair]['tokenizer'] is None:
        model_name = MODELS[lang_pair]['model_name']
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            MODELS[lang_pair]['tokenizer'] = tokenizer
            MODELS[lang_pair]['model'] = model
        except Exception as e:
            print(f"Could not load model for {lang_pair}: {e}")
            return None, None
    
    return MODELS[lang_pair]['tokenizer'], MODELS[lang_pair]['model']

def translate(text, target_lang='hi'):
    """Translate text to target language"""
    if target_lang == 'en':
        return text 
    
    lang_pair = f'en-{target_lang}'
    tokenizer, model = load_model(lang_pair)
    
    if tokenizer is None or model is None:
        print(f"Translation model not available for {target_lang}, returning original text")
        return text
    
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_tts_audio(text, lang_code='en', method='gtts'):
    """
    Generate TTS audio file from text
    
    Args:
        text: Text to convert to speech
        lang_code: Language code ('en', 'hi', 'ta')
        method: TTS method ('gtts' for Google TTS, 'pyttsx3' for offline TTS)
    
    Returns:
        Path to generated audio file
    """
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()
        
        if method == 'gtts':
            lang_map = {
                'en': 'en',
                'hi': 'hi',
                'ta': 'ta'
            }
            tts = gTTS(text=text, lang=lang_map.get(lang_code, 'en'), slow=False)
            tts.save(temp_path)
            
        elif method == 'pyttsx3':
            # Use offline TTS (pyttsx3)
            engine = pyttsx3.init()
            
            # Set properties for different languages
            voices = engine.getProperty('voices')
            
            # Try to find appropriate voice for language
            selected_voice = None
            for voice in voices:
                if lang_code == 'hi' and ('hindi' in voice.name.lower() or 'devanagari' in voice.name.lower()):
                    selected_voice = voice.id
                    break
                elif lang_code == 'ta' and ('tamil' in voice.name.lower()):
                    selected_voice = voice.id
                    break
                elif lang_code == 'en' and ('english' in voice.name.lower() or 'en' in voice.id.lower()):
                    selected_voice = voice.id
                    break
            
            if selected_voice:
                engine.setProperty('voice', selected_voice)
            
            # Set speech rate and volume
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            # Save to file
            engine.save_to_file(text, temp_path.replace('.mp3', '.wav'))
            engine.runAndWait()
            
            # Update path for wav file
            temp_path = temp_path.replace('.mp3', '.wav')
        
        return temp_path
        
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None

ONNX_PATH = os.path.join(os.getcwd(), "model", "plant_leaf_diseases_model.onnx")
if not os.path.isfile(ONNX_PATH):
    raise RuntimeError(f"Model file not found at {ONNX_PATH}")

sess_opts = ort.SessionOptions()
session  = ort.InferenceSession(ONNX_PATH, sess_opts, providers=["CPUExecutionProvider"])
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
}

class CustomHTTPChatModel(BaseChatModel):
    api_key: str          = Field(...)
    base_url: str         = Field(...)
    workspace_slug: str   = Field(...)

    def _call_api(self, message: str) -> str:
        url = f"{self.base_url}/workspace/{self.workspace_slug}/chat"
        headers = {
            "accept":        "application/json",
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "message":     message,
            "mode":        "chat",
            "sessionId":   "example-session-id",
            "attachments": []
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=STREAM_TIMEOUT+5)
        resp.raise_for_status()
        return resp.json().get("textResponse", "")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop:     List[str]                          = None,
        run_manager: CallbackManagerForLLMRun        = None,
        **kwargs: Any
    ) -> ChatResult:
        user_text     = messages[-1].content
        response_text = self._call_api(user_text)
        gen = ChatGeneration(
            message=AIMessage(content=response_text),
            text=response_text
        )
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "custom-http-chat"


@app.route("/analyze", methods=["POST"])
@cross_origin()
def analyze():
    try:
        if "file" not in request.files or "lang" not in request.form:
            return jsonify({"error": "file and lang are required"}), 400

        lang_code = request.form["lang"].strip()
        lang_name = LANG_MAP.get(lang_code, "English")
        img_bytes = request.files["file"].read()

        data       = preprocess_image(img_bytes)
        raw_output = session.run([output_name], {input_name: data})[0].flatten()
        if raw_output.shape[0] != len(CLASS_NAMES):
            return jsonify({
                "error":      "Model output length mismatch",
                "output_len": int(raw_output.shape[0]),
                "num_classes": len(CLASS_NAMES)
            }), 500

        top_idx  = int(np.argmax(raw_output))
        top3_idx = np.argsort(raw_output)[-3:][::-1]
        predicted  = CLASS_NAMES[top_idx]
        confidence = float(raw_output[top_idx] * 100)
        top3 = [
            {"label": CLASS_NAMES[i], "score": float(raw_output[i] * 100)}
            for i in top3_idx
        ]
        plant = predicted.split('_')[0]

        # Get LLM response in English first
        full_prompt = (
            f"Plant: {plant}\n"
            f"Disease: {predicted}\n\n"
            "Respond ONLY with a raw JSON object. Do not include explanations or extra text.\n"
            "The JSON must have these keys:\n"
            "  disease_type, symptoms, prevention, treatments, fertilizers, expected_yield\n"
            "Please respond in English."
        )

        chat = CustomHTTPChatModel(
            api_key       = API_KEY,
            base_url      = BASE_URL,
            workspace_slug= WORKSPACE_SLUG
        )
        from langchain_core.messages import HumanMessage
        llm_result = chat.generate(
        [[HumanMessage(content=full_prompt)]],
        stop=[]
        )
        first_gen: ChatGeneration = llm_result.generations[0][0]
        response_text = first_gen.text
        llm_json = json.loads(response_text)

        # Translate each field if language is not English
        if lang_code != 'en':
            translated_fields = {}
            for key, value in llm_json.items():
                if value and isinstance(value, str):
                    translated_fields[key] = translate(value, lang_code)
                else:
                    translated_fields[key] = value
            llm_json = translated_fields

        # Create TTS message
        disease_name = predicted.replace('_', ' ')
        tts_message = f"Detected plant {plant}. Disease: {disease_name} with confidence {confidence:.1f} percent."
        
        # Translate TTS message if language is not English
        if lang_code != 'en':
            tts_message = translate(tts_message, lang_code)

        # Generate TTS audio file
        audio_file_path = None
        tts_method = request.form.get("tts_method", "gtts")  # Default to Google TTS
        
        if request.form.get("generate_tts", "false").lower() == "true":
            audio_file_path = generate_tts_audio(tts_message, lang_code, tts_method)

        response_data = {
            "predicted":      predicted,
            "confidence":     confidence,
            "top3":           top3,
            "disease_type":   llm_json.get("disease_type", ""),
            "symptoms":       llm_json.get("symptoms", ""),
            "prevention":     llm_json.get("prevention", ""),
            "treatments":     llm_json.get("treatments", ""),
            "fertilizers":    llm_json.get("fertilizers", ""),
            "expected_yield": llm_json.get("expected_yield", ""),
            "tts_message":    tts_message,
            "tts_available":  audio_file_path is not None
        }

        if audio_file_path:
            response_data["tts_audio_url"] = f"/get_audio/{os.path.basename(audio_file_path)}"

        return jsonify(response_data)

    except requests.RequestException as re:
        app.logger.error("LLM API error: %s", re)
        return jsonify({"error": "LLM API request failed", "details": str(re)}), 502
    except Exception as e:
        app.logger.exception("Error in /analyze")
        return jsonify({"error": str(e)}), 500


@app.route("/get_audio/<filename>")
@cross_origin()
def get_audio(filename):
    """Serve generated TTS audio files"""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "Audio file not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate_tts", methods=["POST"])
@cross_origin()
def generate_tts_endpoint():
    """Separate endpoint for generating TTS from text"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "text is required"}), 400
        
        text = data["text"]
        lang_code = data.get("lang", "en")
        tts_method = data.get("method", "gtts")
        
        audio_file_path = generate_tts_audio(text, lang_code, tts_method)
        
        if audio_file_path:
            return jsonify({
                "success": True,
                "audio_url": f"/get_audio/{os.path.basename(audio_file_path)}"
            })
        else:
            return jsonify({"error": "Failed to generate TTS audio"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)