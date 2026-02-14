import React, { useState, useRef } from 'react';
import {
  Box,
  Button,
  IconButton,
  Typography,
  LinearProgress,
  Alert,
  Chip,
  TextField,
} from '@mui/material';
import {
  Mic as MicIcon,
  Stop as StopIcon,
  Delete as DeleteIcon,
  Send as SendIcon,
} from '@mui/icons-material';
import axios from 'axios';

function VoiceInput({ onTranscriptionComplete, inputType = "chief_complaint" }) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [detectedLanguage, setDetectedLanguage] = useState("");
  const [error, setError] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError("Could not access microphone. Please check permissions.");
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const transcribeAudio = async () => {
    if (!audioBlob) return;

    setIsTranscribing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('input_type', inputType);

      const response = await axios.post('/api/transcribe', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setTranscription(response.data.text);
        setDetectedLanguage(response.data.language);
        
        // Call parent callback with transcription
        if (onTranscriptionComplete) {
          onTranscriptionComplete(response.data.text);
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Transcription failed");
      console.error(err);
    } finally {
      setIsTranscribing(false);
    }
  };

  const clearRecording = () => {
    setAudioBlob(null);
    setTranscription("");
    setDetectedLanguage("");
    setError(null);
    audioChunksRef.current = [];
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="subtitle2" gutterBottom>
        Voice Input (99 languages supported)
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        {!isRecording && !audioBlob && (
          <Button
            variant="outlined"
            startIcon={<MicIcon />}
            onClick={startRecording}
            color="primary"
          >
            Start Recording
          </Button>
        )}

        {isRecording && (
          <Button
            variant="contained"
            startIcon={<StopIcon />}
            onClick={stopRecording}
            color="error"
          >
            Stop Recording
          </Button>
        )}

        {audioBlob && !isTranscribing && (
          <>
            <Button
              variant="contained"
              startIcon={<SendIcon />}
              onClick={transcribeAudio}
              color="primary"
            >
              Transcribe
            </Button>
            <IconButton onClick={clearRecording} color="error">
              <DeleteIcon />
            </IconButton>
          </>
        )}

        {detectedLanguage && (
          <Chip
            label={`Detected: ${detectedLanguage.toUpperCase()}`}
            color="success"
            size="small"
          />
        )}
      </Box>

      {isRecording && (
        <Box sx={{ mb: 2 }}>
          <LinearProgress color="error" />
          <Typography variant="caption" color="error">
            Recording... Speak now
          </Typography>
        </Box>
      )}

      {isTranscribing && (
        <Box sx={{ mb: 2 }}>
          <LinearProgress />
          <Typography variant="caption">
            Transcribing with Whisper (local model)...
          </Typography>
        </Box>
      )}

      {transcription && (
        <TextField
          fullWidth
          multiline
          rows={3}
          value={transcription}
          onChange={(e) => setTranscription(e.target.value)}
          label="Transcribed Text"
          variant="outlined"
          sx={{ mt: 1 }}
        />
      )}

      <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
        Powered by OpenAI Whisper (local) - Supports 99 languages including Hindi, Bengali, Tamil, Telugu, etc.
      </Typography>
    </Box>
  );
}

export default VoiceInput;
