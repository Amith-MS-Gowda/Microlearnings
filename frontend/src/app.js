import React, { useState } from "react";
import axios from "axios";
import './app.css'; // Importing the external CSS file

function App() {
  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSummarizeText = async () => {
    if (!text) {
      setSummary("Please provide some text to summarize.");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:5000/summarize_text", { text });
      setSummary(response.data.summary || "Error summarizing text");
    } catch (error) {
      console.error(error);
      setSummary("An error occurred during text summarization.");
    } finally {
      setLoading(false);
    }
  };

  const handleAudioUpload = async () => {
    if (!audioFile) {
      setSummary("Please upload an audio file to summarize.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("audio", audioFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/summarize_audio", formData);
      setSummary(response.data.summary || "Error summarizing audio");
    } catch (error) {
      console.error(error);
      setSummary("An error occurred during audio summarization.");
    } finally {
      setLoading(false);
    }
  };

  const handleVideoUpload = async () => {
    if (!videoFile) {
      setSummary("Please upload a video file to summarize.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/summarize_video", formData);
      setSummary(response.data.message || "Video summarized successfully. File saved.");
    } catch (error) {
      console.error(error);
      setSummary("An error occurred during video summarization.");
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async () => {
    if (!imageFile) {
      setSummary("Please upload an image file to summarize.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/summarize_image", formData);
      setSummary(response.data.summary || "Error summarizing image");
    } catch (error) {
      console.error(error);
      setSummary("An error occurred during image summarization.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Microlearning Summarizer</h1>
      </header>

      <div className="content-container">
        <div className="card">
          <h2>Text Summarizer</h2>
          <textarea
            className="textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your text here..."
          />
          <br />
          <button
            onClick={handleSummarizeText}
            className={`btn ${loading ? 'btn-disabled' : ''}`}
            disabled={loading}
          >
            {loading ? "Summarizing..." : "Summarize Text"}
          </button>
        </div>

        <div className="card">
          <h2>Audio Summarizer</h2>
          <input
            type="file"
            onChange={(e) => setAudioFile(e.target.files[0])}
            accept="audio/*"
            className="file-upload"
          />
          <br />
          <button
            onClick={handleAudioUpload}
            className={`btn ${loading ? 'btn-disabled' : ''}`}
            disabled={loading}
          >
            {loading ? "Summarizing..." : "Summarize Audio"}
          </button>
        </div>

        <div className="card">
          <h2>Video Summarizer</h2>
          <input
            type="file"
            onChange={(e) => setVideoFile(e.target.files[0])}
            accept="video/*"
            className="file-upload"
          />
          <br />
          <button
            onClick={handleVideoUpload}
            className={`btn ${loading ? 'btn-disabled' : ''}`}
            disabled={loading}
          >
            {loading ? "Summarizing..." : "Summarize Video"}
          </button>
        </div>

        <div className="card">
          <h2>Image Summarizer</h2>
          <input
            type="file"
            onChange={(e) => setImageFile(e.target.files[0])}
            accept="image/*"
            className="file-upload"
          />
          <br />
          <button
            onClick={handleImageUpload}
            className={`btn ${loading ? 'btn-disabled' : ''}`}
            disabled={loading}
          >
            {loading ? "Summarizing..." : "Summarize Image"}
          </button>
        </div>
      </div>

      {loading && <p className="loading-text">Processing your request...</p>}

      <div className="summary-box">
        <h3>Summary</h3>
        <p>{summary}</p>
      </div>

      <footer className="app-footer">
        <p>&copy; 2024 Microlearning Summarizer. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
