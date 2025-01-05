import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
import time
import hashlib
from fpdf import FPDF
from datetime import datetime

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAIO - Projeto de Processamento de Áudio com IA")
        self.root.geometry("1200x800")
        
        # Initialize Whisper model
        self.model = whisper.load_model("medium")
        
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
            ],
            "Violência": [
                "agressão", "briga", "pancada", "espancamento", "soco",
                "chute", "arma", "facada", "tiro", "ameaça"
            ],
            "Dinheiro": [
                "roubo", "fraude", "propina", "suborno", "extorsão",
                "lavagem", "desvio", "corrupção", "sonegação", "golpe"
            ],
            "Armas": [
                "revólver", "pistola", "fuzil", "metralhadora", "munição",
                "explosivo", "granada", "bomba", "armamento", "calibre"
            ],
            "Tráfico Humano": [
                "escravidão", "exploração", "sequestro", "cárcere", "prostituição",
                "aliciamento", "contrabando", "coação", "trabalho forçado", "servidão"
            ],
            "Discriminação": [
                "racismo", "homofobia", "preconceito", "xenofobia", "intolerância",
                "machismo", "segregação", "bullying", "injúria", "difamação"
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
            "Palavras Ofensivas": tk.BooleanVar(value=True),
            "Violência": tk.BooleanVar(value=True),
            "Dinheiro": tk.BooleanVar(value=True),
            "Armas": tk.BooleanVar(value=True),
            "Tráfico Humano": tk.BooleanVar(value=True),
            "Discriminação": tk.BooleanVar(value=True)
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
        
        # Play/Stop button
        self.play_button = ctk.CTkButton(
            self.controls_frame,
            text="▶ Reproduzir",
            command=self.toggle_playback,
            state="disabled"
        )
        self.play_button.pack(pady=5)
        
        # Process button
        self.process_button = ctk.CTkButton(
            self.controls_frame,
            text="Processar Áudio",
            command=self.process_audio,
            state="disabled"
        )
        self.process_button.pack(pady=5)
        
        # Botão de busca de temas
        self.search_button = ctk.CTkButton(
            self.controls_frame,
            text="Buscar Temas",
            command=self.search_sensitive_topics,
            state="disabled"
        )
        self.search_button.pack(pady=5)
        
        # Botão para gerar relatório PDF
        self.pdf_button = ctk.CTkButton(
            self.controls_frame,
            text="Gerar Relatório PDF",
            command=self.generate_pdf_report,
            state="disabled"
        )
        self.pdf_button.pack(pady=5)
        
        # Topics frame with scrollbar
        self.topics_frame = ctk.CTkScrollableFrame(self.left_panel)
        self.topics_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        topics_label = ctk.CTkLabel(
            self.topics_frame,
            text="Temas Sensíveis:"
        )
        topics_label.pack(pady=5)
        
        # Create checkboxes for each topic
        for topic in sorted(self.topics.keys()):
            ctk.CTkCheckBox(
                self.topics_frame,
                text=topic,
                variable=self.topics[topic]
            ).pack(pady=2, anchor=tk.W, padx=5)
        
        # Custom topic entry
        self.custom_topic = ctk.CTkEntry(
            self.topics_frame,
            placeholder_text="Adicionar novo tema..."
        )
        self.custom_topic.pack(pady=5, padx=5, fill=tk.X)
        
        add_topic_button = ctk.CTkButton(
            self.topics_frame,
            text="Adicionar Tema",
            command=self.add_custom_topic
        )
        add_topic_button.pack(pady=5)
        
        # Right panel for visualization and results
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Waveform plot
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame para a transcrição com scrollbar
        transcription_label = ctk.CTkLabel(self.right_panel, text="Transcrição:")
        transcription_label.pack(pady=5)
        
        self.transcription_frame = ctk.CTkScrollableFrame(self.right_panel, height=200)
        self.transcription_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.transcription_text = ctk.CTkFrame(self.transcription_frame)
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        
        # Frame para os resultados da busca com scrollbar
        results_label = ctk.CTkLabel(self.right_panel, text="Temas Sensíveis Detectados:")
        results_label.pack(pady=5)
        
        self.results_frame = ctk.CTkScrollableFrame(self.right_panel, height=200)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.matches_text = ctk.CTkTextbox(self.results_frame, wrap=tk.WORD)
        self.matches_text.pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg")]
        )
        if self.filename:
            self.load_audio_visualization()
            self.load_audio_playback()
            self.play_button.configure(state="normal")
            self.process_button.configure(state="normal")
    
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
    
    def toggle_playback(self):
        """Toggle between play and stop"""
        if self.is_playing:
            self.stop_playback()
            self.play_button.configure(text="▶ Reproduzir")
        else:
            self.start_playback()
            self.play_button.configure(text="⏹ Parar")
    
    def start_playback(self):
        """Start audio playback"""
        if not self.audio_segment or self.is_playing:
            return
            
        self.is_playing = True
        self.stop_requested = False
        
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
            self.root.after(0, lambda: self.play_button.configure(text="▶ Reproduzir"))
    
    def add_custom_topic(self):
        new_topic = self.custom_topic.get()
        if new_topic and new_topic not in self.topics:
            self.topics[new_topic] = tk.BooleanVar(value=True)
            self.topic_related_words[new_topic] = [new_topic]  # Usa o próprio tema como palavra relacionada
            
            # Remove todos os widgets do frame
            for widget in self.topics_frame.winfo_children():
                widget.destroy()
            
            # Recria o título
            topics_label = ctk.CTkLabel(
                self.topics_frame,
                text="Temas Sensíveis:"
            )
            topics_label.pack(pady=5)
            
            # Recria todos os checkboxes em ordem alfabética
            for topic in sorted(self.topics.keys()):
                ctk.CTkCheckBox(
                    self.topics_frame,
                    text=topic,
                    variable=self.topics[topic]
                ).pack(pady=2, anchor=tk.W, padx=5)
            
            # Recria o campo de entrada e botões
            self.custom_topic = ctk.CTkEntry(
                self.topics_frame,
                placeholder_text="Adicionar novo tema..."
            )
            self.custom_topic.pack(pady=5, padx=5, fill=tk.X)
            
            add_topic_button = ctk.CTkButton(
                self.topics_frame,
                text="Adicionar Tema",
                command=self.add_custom_topic
            )
            add_topic_button.pack(pady=5)
    
    def play_segment(self, start_time, end_time):
        """Reproduz um segmento específico do áudio"""
        if self.audio_segment:
            # Para qualquer reprodução em andamento
            self.stop_playback()
            
            # Extrai o segmento desejado
            segment = self.audio_segment[int(start_time * 1000):int(end_time * 1000)]
            
            # Salva temporariamente o segmento
            temp_file = "temp_segment.wav"
            segment.export(temp_file, format="wav")
            
            # Carrega e reproduz o segmento
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Remove o arquivo temporário após um curto delay
            def cleanup():
                time.sleep((end_time - start_time) + 0.5)  # Espera o áudio terminar
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            threading.Thread(target=cleanup, daemon=True).start()

    def create_sentence_frame(self, text, start_time, end_time):
        """Cria um frame para uma sentença com botão de reprodução"""
        frame = ctk.CTkFrame(self.transcription_text)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Botão de reprodução
        play_btn = ctk.CTkButton(
            frame,
            text="▶",
            width=30,
            command=lambda s=start_time, e=end_time: self.play_segment(s, e)
        )
        play_btn.pack(side=tk.LEFT, padx=5)
        
        # Texto da transcrição
        text_label = ctk.CTkLabel(
            frame,
            text=f"[{start_time:.2f}s - {end_time:.2f}s] {text}",
            wraplength=500,
            justify=tk.LEFT
        )
        text_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        return frame

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
        
        try:
            # Clear previous results
            for widget in self.transcription_text.winfo_children():
                widget.destroy()
            self.matches_text.delete("1.0", tk.END)

            # Load and process audio file
            result = self.model.transcribe(self.filename)
            
            # Store sentences
            self.sentences = [(segment["text"].strip(), segment["start"], segment["end"]) 
                            for segment in result["segments"]]

            # Display transcription with segments
            for text, start, end in self.sentences:
                self.create_sentence_frame(text, start, end)

            # Enable search and PDF buttons
            self.search_button.configure(state="normal")
            self.pdf_button.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar o áudio: {str(e)}")

    def search_sensitive_topics(self):
        """Search for sensitive topics in already transcribed sentences"""
        if not hasattr(self, 'sentences'):
            messagebox.showerror("Erro", "Por favor, processe um áudio primeiro.")
            return
        
        # Clear previous results
        self.matches_text.delete("1.0", tk.END)
        
        # Check each sentence for sensitive topics
        matches = []
        for text, start, end in self.sentences:
            for topic, var in self.topics.items():
                if var.get():  # Se o tema está selecionado
                    is_sensitive, similar_words = self.find_sensitive_content(text, topic)
                    if is_sensitive:
                        # Get the related words that were used in detection
                        related_words = ", ".join(self.topic_related_words[topic])
                        detected_words = ", ".join(similar_words)
                        
                        matches.append(
                            f"Tema '{topic}' detectado na frase:\n"
                            f"[{start:.2f}s - {end:.2f}s] '{text}'\n"
                            f"Palavras relacionadas ao tema: {related_words}\n"
                            f"Palavras detectadas: {detected_words}\n\n"
                        )
        
        # Display matches
        if matches:
            self.matches_text.insert("1.0", "".join(matches))
        else:
            self.matches_text.insert("1.0", "Nenhum tema sensível detectado.")
    
    def calculate_file_hash(self, filepath):
        """Calcula o hash SHA-256 do arquivo"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Ler o arquivo em blocos para não sobrecarregar a memória
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def generate_pdf_report(self):
        """Gera um relatório PDF com a transcrição e análise de temas sensíveis"""
        if not hasattr(self, 'sentences'):
            messagebox.showerror("Erro", "Nenhuma transcrição disponível para gerar relatório.")
            return
            
        # Solicita local para salvar o PDF
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Salvar Relatório PDF"
        )
        
        if not file_path:
            return
            
        try:
            # Calcular hash do arquivo
            file_hash = self.calculate_file_hash(self.filename)
            
            # Criar PDF
            pdf = FPDF()
            
            # Configurar página
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_margins(20, 20, 20)
            
            # Título
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Relatório de Análise de Áudio', ln=True, align='C')
            pdf.ln(5)
            
            # Informações do arquivo
            pdf.set_font('Arial', '', 10)
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            pdf.cell(0, 6, f'Data: {current_time}', ln=True)
            pdf.cell(0, 6, f'Arquivo: {os.path.basename(self.filename)}', ln=True)
            
            # Hash do arquivo
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, 'Hash SHA-256:', ln=True)
            pdf.set_font('Arial', '', 8)  # Fonte menor para o hash
            pdf.cell(0, 6, file_hash, ln=True)
            pdf.ln(5)
            
            # Transcrição
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Transcrição:', ln=True)
            
            pdf.set_font('Arial', '', 10)
            for text, start, end in self.sentences:
                timestamp = f'[{start:.2f}s - {end:.2f}s]'
                pdf.multi_cell(0, 6, f'{timestamp}\n{text}', ln=True)
                pdf.ln(2)
            
            # Temas Sensíveis
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Temas Sensíveis Detectados:', ln=True)
            
            # Procura por temas sensíveis
            found_topics = False
            pdf.set_font('Arial', '', 10)
            
            for text, start, end in self.sentences:
                for topic, var in self.topics.items():
                    if var.get():  # Se o tema está selecionado
                        has_content, similar_words = self.find_sensitive_content(text, topic)
                        if has_content:
                            found_topics = True
                            timestamp = f'[{start:.2f}s - {end:.2f}s]'
                            pdf.set_font('Arial', 'B', 10)
                            pdf.cell(0, 6, f'Tema: {topic}', ln=True)
                            pdf.set_font('Arial', '', 10)
                            pdf.multi_cell(0, 6, f'Tempo: {timestamp}\nTexto: {text}\nPalavras detectadas: {", ".join(similar_words)}', ln=True)
                            pdf.ln(5)
            
            if not found_topics:
                pdf.multi_cell(0, 6, 'Nenhum tema sensível detectado.', ln=True)
            
            # Salvar PDF
            pdf.output(file_path)
            
            # Mostrar mensagem de sucesso
            messagebox.showinfo("Sucesso", f"Relatório PDF gerado com sucesso!\nSalvo em: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar PDF: {str(e)}")

if __name__ == "__main__":
    root = ctk.CTk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
