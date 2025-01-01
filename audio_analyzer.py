import os
import tkinter as tk
from tkinter import ttk, filedialog
import customtkinter as ctk
import whisper
import spacy
import nltk
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydub import AudioSegment
from pydub.playback import play
import threading

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")
        self.root.geometry("1200x800")
        
        # Initialize Whisper model
        self.model = whisper.load_model("base")
        
        # Initialize Spacy
        self.nlp = spacy.load("pt_core_news_sm")
        
        # Download necessary NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # Audio playback variables
        self.audio_segment = None
        self.is_playing = False
        self.play_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # File selection and playback controls
        self.controls_frame = ctk.CTkFrame(self.left_panel)
        self.controls_frame.pack(pady=10, fill=tk.X)
        
        self.select_button = ctk.CTkButton(
            self.controls_frame,
            text="Selecionar Arquivo de Áudio",
            command=self.select_file
        )
        self.select_button.pack(pady=5)
        
        self.play_button = ctk.CTkButton(
            self.controls_frame,
            text="▶ Reproduzir",
            command=self.toggle_play,
            state="disabled"
        )
        self.play_button.pack(pady=5)
        
        # Sensitive topics frame
        self.topics_frame = ctk.CTkFrame(self.left_panel)
        self.topics_frame.pack(pady=10, fill=tk.X)
        
        ctk.CTkLabel(self.topics_frame, text="Temas Sensíveis:").pack()
        
        # Default sensitive topics
        self.topics = {
            "Drogas": tk.BooleanVar(value=True),
            "Morte": tk.BooleanVar(value=True),
            "Crimes Sexuais": tk.BooleanVar(value=True)
        }
        
        for topic, var in self.topics.items():
            ctk.CTkCheckBox(
                self.topics_frame,
                text=topic,
                variable=var
            ).pack(pady=2)
        
        # Custom topic entry
        self.custom_topic = ctk.CTkEntry(
            self.topics_frame,
            placeholder_text="Adicionar novo tema..."
        )
        self.custom_topic.pack(pady=5)
        
        ctk.CTkButton(
            self.topics_frame,
            text="Adicionar Tema",
            command=self.add_custom_topic
        ).pack(pady=5)
        
        # Process button
        self.process_button = ctk.CTkButton(
            self.left_panel,
            text="Processar Áudio",
            command=self.process_audio
        )
        self.process_button.pack(pady=10)
        
        # Right panel for visualization
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Waveform plot
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Transcription text
        self.transcription_text = ctk.CTkTextbox(
            self.right_panel,
            height=300
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def select_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg")]
        )
        if self.filename:
            self.load_audio_visualization()
            self.load_audio_playback()
            self.play_button.configure(state="normal")
    
    def load_audio_visualization(self):
        # Load audio file
        y, sr = librosa.load(self.filename)
        
        # Clear previous plot
        self.ax.clear()
        
        # Plot waveform
        self.ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Waveform')
        
        # Update canvas
        self.canvas.draw()
    
    def load_audio_playback(self):
        """Load audio file for playback"""
        try:
            self.audio_segment = AudioSegment.from_file(self.filename)
        except Exception as e:
            print(f"Error loading audio: {e}")
            self.audio_segment = None
    
    def toggle_play(self):
        """Toggle audio playback"""
        if not self.audio_segment:
            return
            
        if self.is_playing:
            self.is_playing = False
            self.play_button.configure(text="▶ Reproduzir")
        else:
            self.is_playing = True
            self.play_button.configure(text="⏸ Pausar")
            
            # Start playback in a separate thread to avoid blocking the UI
            self.play_thread = threading.Thread(target=self.play_audio)
            self.play_thread.daemon = True
            self.play_thread.start()
    
    def play_audio(self):
        """Play audio in a separate thread"""
        try:
            play(self.audio_segment)
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            self.is_playing = False
            self.play_button.configure(text="▶ Reproduzir")
            
    def add_custom_topic(self):
        new_topic = self.custom_topic.get()
        if new_topic and new_topic not in self.topics:
            self.topics[new_topic] = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                self.topics_frame,
                text=new_topic,
                variable=self.topics[new_topic]
            ).pack(pady=2)
            self.custom_topic.delete(0, tk.END)
    
    def process_audio(self):
        if not hasattr(self, 'filename'):
            return
        
        # Transcribe audio
        result = self.model.transcribe(self.filename)
        transcription = result["text"]
        
        # Process text for sensitive topics
        doc = self.nlp(transcription)
        
        # Highlight sensitive topics
        highlighted_text = transcription
        for topic, var in self.topics.items():
            if var.get():
                # Case-insensitive search
                topic_lower = topic.lower()
                text_lower = highlighted_text.lower()
                
                # Find all occurrences and highlight them
                start = 0
                while True:
                    pos = text_lower.find(topic_lower, start)
                    if pos == -1:
                        break
                    
                    # Add highlighting markers
                    highlighted_text = (
                        highlighted_text[:pos] +
                        "**" + highlighted_text[pos:pos+len(topic)] + "**" +
                        highlighted_text[pos+len(topic):]
                    )
                    
                    # Update search position
                    start = pos + len(topic) + 4  # +4 for the added ** markers
        
        # Display results
        self.transcription_text.delete("1.0", tk.END)
        self.transcription_text.insert("1.0", highlighted_text)

if __name__ == "__main__":
    root = ctk.CTk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
