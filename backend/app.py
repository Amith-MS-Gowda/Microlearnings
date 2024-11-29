from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BartTokenizer, BartForConditionalGeneration
import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from PIL import Image
import torch
from torchvision import models, transforms
import requests
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained BART model and tokenizer for text summarization
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Image classification setup (ResNet model)
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Image preprocessing pipeline for ResNet50
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define ImageNet class labels (top-1000 most common objects)
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
labels = requests.get(LABELS_URL).json()
class_idx = {int(key): value[1] for key, value in labels.items()}

# Function to split large text into chunks of a maximum number of tokens for summarization
def split_text_into_chunks(text, chunk_size=1024):
    words = text.split()
    chunks = []
    chunk = []
    
    for word in words:
        chunk.append(word)
        if len(" ".join(chunk)) > chunk_size:
            chunks.append(" ".join(chunk[:-1]))
            chunk = [word]
    
    if chunk:
        chunks.append(" ".join(chunk))
    
    return chunks

# Function to summarize text
def summarize_text(text):
    chunks = split_text_into_chunks(text)
    summaries = []
    for chunk in chunks:
        try:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=150, 
                num_beams=4, 
                length_penalty=2.0, 
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            return {"error": f"Error during text summarization: {str(e)}"}

    # Combine the summaries of all chunks
    return " ".join(summaries)

# Route: Text Summarization (Using BART model)
@app.route('/summarize_text', methods=['POST'])
def summarize_text_route():
    # Get the input text from the POST request
    data = request.get_json()
    text = data.get("text", "")
    
    # Validate input
    if not text:
        return jsonify({"summary": "No text provided"}), 400

    summary = summarize_text(text)
    return jsonify({"summary": summary})

# Route: Video Summarization (Convert Audio to Text and Summarize)
@app.route('/summarize_video', methods=['POST'])
def summarize_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    
    # Validate video file format
    if not video_file.filename.lower().endswith(('mp4', 'avi', 'mov')):
        return jsonify({"error": "Invalid video file format. Please upload an MP4, AVI, or MOV file."}), 400

    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    try:
        # Load the video file
        clip = VideoFileClip(video_path)
        audio_path = os.path.join("uploads", "audio_from_video.wav")
        clip.audio.write_audiofile(audio_path)  # Extract audio from video

        # Convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Summarize the text from audio
        summary = summarize_text(text)

        return jsonify({"summary": summary})
    
    except Exception as e:
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500
    finally:
        os.remove(video_path)
        os.remove(audio_path)

# Route: Audio Summarization (Speech to Text and Summarize)
@app.route('/summarize_audio', methods=['POST'])
def summarize_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    # Validate audio file format
    if not audio_file.filename.lower().endswith(('wav', 'mp3', 'flac')):
        return jsonify({"error": "Invalid audio file format. Please upload a WAV, MP3, or FLAC file."}), 400

    audio_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    try:
        # Recognize speech from the audio file
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Summarize the recognized text
        summary = summarize_text(text)

        return jsonify({"summary": summary})
    
    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
    finally:
        os.remove(audio_path)

# Route: Image Summarization (Object Detection and Descriptive Summary)
@app.route('/summarize_image', methods=['POST'])
def summarize_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = os.path.join("uploads", image_file.filename)
    image_file.save(image_path)

    try:
        # Open the image file
        image = Image.open(image_path)
        
        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        # Make sure we are on the right device (CPU or GPU)
        with torch.no_grad():
            output = resnet_model(input_batch)

        # Get the top predicted class index
        _, predicted_class = torch.max(output, 1)
        predicted_label = class_idx[predicted_class.item()]

        # Generate a descriptive summary of the image
        description = f"The image contains a {predicted_label}."
        summary = summarize_text(description)  # Summarize the description

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": f"Error during image summarization: {str(e)}"}), 500
    finally:
        os.remove(image_path)


if __name__ == "__main__":
    app.run(debug=True)
