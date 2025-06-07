import pyttsx3
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set properties for the audio
engine.setProperty("rate", 160)  # Speed of speech
engine.setProperty("volume", 0.9)  # Volume (0.0 to 1.0)

# Sample text to convert to audio
sample_text = "What are the action items from the recent meetings?"

# Save to WAV file
output_file = "sample_meeting_query.wav"
engine.save_to_file(sample_text, output_file)
engine.runAndWait()

print(f"Sample audio file created: {output_file}")