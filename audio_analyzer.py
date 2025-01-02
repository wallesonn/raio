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
import pygame
import io
import threading

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")
        self.root.geometry("1200x800")
        
        # Initialize Whisper model
        self.model = whisper.load_model("base")
        
        # Initialize Spacy with medium model that includes word vectors
        self.nlp = spacy.load("pt_core_news_md")
        
        # Similarity threshold for topic detection
        self.similarity_threshold = 0.5
        
        # Dictionary of related words for each topic
        self.topic_related_words = {
            "Drogas": [
                "cocaína", "maconha", "crack", "heroína", "tráfico",
                "vício", "substância", "entorpecente", "dependente", "overdose"
            ],
            "Morte": [
                "falecimento", "assassinato", "homicídio", "suicídio", "funeral",
                "velório", "cemitério", "luto", "óbito", "cadáver"
            ],
            "Crimes Sexuais": [
                "estupro", "abuso", "assédio", "pedofilia", "violência",
                "exploração", "atentado", "violação", "molestamento", "agressão"
            ],
            "Família": [
                "pai", "mãe", "filho", "irmão", "parente",
                "casamento", "divórcio", "adoção", "guarda", "pensão"
            ],
            "Palavras Ofensivas": [
                "merda", "porra", "caralho", "puta", "viado",
                "buceta", "idiota", "imbecil", "babaca", "cuzão"
            ]
        }
        
        # Download necessary NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Audio playback variables
        self.audio_segment = None
        self.is_playing = False
        self.play_thread = None
        self.stop_requested = False
        
        # Sensitive topics
        self.topics = {
            "Drogas": tk.BooleanVar(value=True),
            "Morte": tk.BooleanVar(value=True),
            "Crimes Sexuais": tk.BooleanVar(value=True),
            "Família": tk.BooleanVar(value=True),
            "Palavras Ofensivas": tk.BooleanVar(value=True)
        }
        
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
            command=self.start_playback,
            state="disabled"
        )
        self.play_button.pack(pady=5)
        
        self.stop_button = ctk.CTkButton(
            self.controls_frame,
            text="⏹ Parar",
            command=self.stop_playback,
            state="disabled"
        )
        self.stop_button.pack(pady=5)
        
        # Sensitive topics frame
        self.topics_frame = ctk.CTkFrame(self.left_panel)
        self.topics_frame.pack(pady=10, fill=tk.X)
        
        ctk.CTkLabel(self.topics_frame, text="Temas Sensíveis:").pack(anchor=tk.W, padx=5)
        
        for topic, var in self.topics.items():
            ctk.CTkCheckBox(
                self.topics_frame,
                text=topic,
                variable=var
            ).pack(pady=2, anchor=tk.W, padx=5)
        
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
            height=200
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Sensitive content matches
        ctk.CTkLabel(self.right_panel, text="Trechos Sensíveis Detectados:").pack(anchor=tk.W, padx=5)
        self.matches_text = ctk.CTkTextbox(
            self.right_panel,
            height=100
        )
        self.matches_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def select_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg")]
        )
        if self.filename:
            self.load_audio_visualization()
            self.load_audio_playback()
            self.play_button.configure(state="normal")
            self.stop_button.configure(state="normal")
    
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
            # Convert audio to WAV format using pydub
            self.audio_segment = AudioSegment.from_file(self.filename)
            wav_data = self.audio_segment.export(format="wav")
            
            # Load the audio data into pygame
            pygame.mixer.music.load(wav_data)
        except Exception as e:
            print(f"Error loading audio: {e}")
            self.audio_segment = None
    
    def start_playback(self):
        """Start audio playback"""
        if not self.audio_segment or self.is_playing:
            return
            
        self.is_playing = True
        self.stop_requested = False
        self.play_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # Start playback in a separate thread
        self.play_thread = threading.Thread(target=self.play_audio)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def stop_playback(self):
        """Stop audio playback"""
        if self.is_playing:
            pygame.mixer.music.stop()
            self.stop_requested = True
            self.is_playing = False
            self.play_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
    
    def play_audio(self):
        """Play audio in a separate thread"""
        try:
            pygame.mixer.music.play()
            
            # Monitor playback status
            while pygame.mixer.music.get_busy() and not self.stop_requested:
                pygame.time.Clock().tick(10)
            
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            self.is_playing = False
            self.root.after(0, lambda: self.play_button.configure(state="normal"))
            self.root.after(0, lambda: self.stop_button.configure(state="disabled"))
    
    def add_custom_topic(self):
        new_topic = self.custom_topic.get()
        if new_topic and new_topic not in self.topics:
            self.topics[new_topic] = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                self.topics_frame,
                text=new_topic,
                variable=self.topics[new_topic]
            ).pack(pady=2, anchor=tk.W, padx=5)
            self.custom_topic.delete(0, tk.END)
    
    def find_similar_words(self, text, topic):
        """
        Find words in text that are semantically similar to the topic and its related words.
        Returns a list of similar words and their similarity scores.
        """
        # Process the text with spaCy
        doc = self.nlp(text.lower())
        similar_words = set()  # Using set to avoid duplicates
        
        # Get related words for the topic
        related_words = self.topic_related_words.get(topic, [])
        
        # Process topic and its related words
        topic_tokens = [self.nlp(topic)[0]] + [self.nlp(word)[0] for word in related_words]
        
        # Check similarity with topic and related words
        for token in doc:
            if token.has_vector and not token.is_stop and not token.is_punct:
                # Check similarity with topic and all related words
                for topic_token in topic_tokens:
                    if topic_token.has_vector:
                        similarity = token.similarity(topic_token)
                        if similarity > self.similarity_threshold:
                            similar_words.add(token.text)
                            break  # If we found a match, no need to check other related words
        
        return list(similar_words)

    def find_sensitive_content(self, sentence, topic):
        """
        Check if a sentence contains content similar to a sensitive topic.
        Returns a tuple of (bool, list of similar words) indicating if sensitive content was found.
        """
        similar_words = self.find_similar_words(sentence, topic)
        return len(similar_words) > 0, similar_words

    def process_audio(self):
        if not hasattr(self, 'filename'):
            return
        
        # Transcribe audio
        result = self.model.transcribe(self.filename)
        transcription = result["text"]
        
        # Clear previous results
        self.transcription_text.delete("1.0", tk.END)
        self.matches_text.delete("1.0", tk.END)
        
        # Display transcription
        self.transcription_text.insert("1.0", transcription)
        
        # Process text for sensitive content
        doc = self.nlp(transcription.lower())
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Check each sentence for sensitive topics
        matches = []
        for sentence in sentences:
            for topic, var in self.topics.items():
                if var.get():  # If topic is selected
                    is_sensitive, similar_words = self.find_sensitive_content(sentence, topic)
                    if is_sensitive:
                        # Get the related words that were used in detection
                        related_words = ", ".join(self.topic_related_words[topic])
                        detected_words = ", ".join(similar_words)
                        
                        matches.append(
                            f"Tema '{topic}' detectado na frase:\n"
                            f"'{sentence}'\n"
                            f"Palavras relacionadas ao tema: {related_words}\n"
                            f"Palavras detectadas: {detected_words}\n\n"
                        )
        
        # Display matches
        if matches:
            self.matches_text.insert("1.0", "".join(matches))
        else:
            self.matches_text.insert("1.0", "Nenhum trecho sensível detectado.")

if __name__ == "__main__":
    root = ctk.CTk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
