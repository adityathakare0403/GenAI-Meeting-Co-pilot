import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [query, setQuery] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await axios.post('http://localhost:8000/stt', { query });
      setResponse(res.data);
    } catch (err) {
      setError('Error processing text query: ' + err.message);
    }
    setLoading(false);
  };

  const handleAudioSubmit = async (e) => {
    e.preventDefault();
    if (!audioFile) {
      setError('Please select an audio file.');
      return;
    }
    setLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', audioFile);
    try {
      const res = await axios.post('http://localhost:8000/stt_audio', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResponse(res.data);
    } catch (err) {
      setError('Error processing audio file: ' + err.message);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-4">
      <h1 className="text-3xl font-bold mb-6">GenAI Meeting Co-pilot</h1>
      
      {/* Text Input Form */}
      <form onSubmit={handleTextSubmit} className="w-full max-w-md mb-6">
        <label className="block text-gray-700 mb-2">Enter Query (Simulated STT):</label>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full p-2 border rounded mb-4"
          placeholder="e.g., What are the action items?"
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:bg-blue-300"
        >
          {loading ? 'Processing...' : 'Submit Query'}
        </button>
      </form>

      {/* Audio Upload Form */}
      <form onSubmit={handleAudioSubmit} className="w-full max-w-md mb-6">
        <label className="block text-gray-700 mb-2">Upload Audio (Optional):</label>
        <input
          type="file"
          accept="audio/wav"
          onChange={(e) => setAudioFile(e.target.files[0])}
          className="w-full p-2 border rounded mb-4"
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-green-500 text-white p-2 rounded hover:bg-green-600 disabled:bg-green-300"
        >
          {loading ? 'Processing...' : 'Submit Audio'}
        </button>
      </form>

      {/* Response Display */}
      {error && <p className="text-red-500 mb-4">{error}</p>}
      {response && (
        <div className="w-full max-w-md bg-white p-4 rounded shadow">
          <h2 className="text-xl font-semibold mb-2">Response</h2>
          <p><strong>Query/Transcription:</strong> {response.query || response.transcription}</p>
          <p><strong>AI Response:</strong> {response.response}</p>
          <p><strong>Co-pilot Suggestion:</strong> {response.copilot_suggestion}</p>
          <p><strong>Retrieved Context:</strong> {response.retrieved_context}</p>
        </div>
      )}
    </div>
  );
};

export default App;