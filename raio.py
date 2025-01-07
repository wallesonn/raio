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
import random
import colorsys

class ProgressWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Processando...")
        
        # Centraliza a janela
        window_width = 300
        window_height = 150
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Frame principal
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Label de status
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Iniciando processamento...",
            text_color="white"
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=(10, 20))
        
        # Barra de progresso
        self.progress_bar = ctk.CTkProgressBar(main_frame)
        self.progress_bar.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.progress_bar.set(0)
        
        # Configura a janela
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # Desabilita o botão de fechar
        
    def update_progress(self, value, status):
        self.progress_bar.set(value)
        self.status_label.configure(text=status)
        self.update()

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
        
        # Estrutura para armazenar resultados pré-processados
        self.processed_sentences = []  # Lista de dicionários com informações das sentenças
        
        # Dictionary of related words for each topic
        self.topic_related_words = {
            "Nenhum": [],
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
        
        # Cores para cada tema sensível
        self.topic_colors = {
            "Nenhum": "#808080",      # Cinza
            "Drogas": "#FF1493",        # Rosa escuro
            "Morte": "#4169E1",         # Azul real
            "Crimes Sexuais": "#228B22", # Verde floresta
            "Família": "#8B008B",       # Roxo escuro
            "Palavras Ofensivas": "#DAA520", # Dourado escuro
            "Violência": "#CD5C5C",     # Vermelho indiano
            "Dinheiro": "#4682B4",      # Azul aço
            "Armas": "#FF4500",         # Laranja vermelho
            "Tráfico Humano": "#483D8B", # Azul ardósia escuro
            "Discriminação": "#B22222"   # Vermelho tijolo
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
            "Nenhum": tk.BooleanVar(value=False),  # Começa desmarcado
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
        
        # Botão para gerar relatório PDF
        self.pdf_button = ctk.CTkButton(
            self.controls_frame,
            text="Gerar Relatório PDF",
            command=self.generate_pdf_report,
            state="disabled"
        )
        self.pdf_button.pack(pady=5)
        
        # Create topics frame
        self.topics_frame = ctk.CTkFrame(self.controls_frame)
        self.topics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add title
        topics_label = ctk.CTkLabel(
            self.topics_frame,
            text="Temas Sensíveis:",
            text_color="white"
        )
        topics_label.pack(pady=5)
        
        # Add checkboxes for each topic
        for topic in sorted(self.topics.keys()):
            checkbox = ctk.CTkCheckBox(
                self.topics_frame,
                text=topic,
                variable=self.topics[topic],
                command=self.filter_transcription,
                fg_color=self.topic_colors[topic],
                text_color="white"
            )
            checkbox.pack(anchor=tk.W, padx=5, pady=2)
        
        # Add topic button
        add_topic_button = ctk.CTkButton(
            self.topics_frame,
            text="Adicionar Tema",
            command=self.add_custom_topic,
            text_color="white"
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
        """Add a custom sensitive topic"""
        if not hasattr(self, 'processed_sentences'):
            messagebox.showerror("Erro", "Por favor, processe um áudio primeiro.")
            return
            
        # Create dialog window
        dialog = ctk.CTkInputDialog(
            text="Digite o nome do novo tema sensível:",
            title="Adicionar Tema"
        )
        new_topic = dialog.get_input()
        
        if new_topic and new_topic.strip():
            new_topic = new_topic.strip()
            
            # Check if topic already exists
            if new_topic in self.topics:
                messagebox.showerror("Erro", "Este tema já existe!")
                return
            
            # Add new topic (using the topic itself as the related word)
            self.topics[new_topic] = tk.BooleanVar(value=True)
            self.topic_related_words[new_topic] = [new_topic]
            
            # Generate a new color for the topic
            hue = random.random()
            saturation = 0.7 + random.random() * 0.3  # 0.7-1.0
            value = 0.5 + random.random() * 0.3      # 0.5-0.8
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            self.topic_colors[new_topic] = color
            
            # Create progress window for analysis
            progress_window = ProgressWindow(self.root)
            progress_window.update_progress(0, f"Analisando novo tema: {new_topic}")
            
            try:
                # Analyze all sentences for the new topic
                total_sentences = len(self.processed_sentences)
                for idx, sentence_data in enumerate(self.processed_sentences):
                    # Update progress
                    progress = (idx + 1) / total_sentences
                    progress_window.update_progress(
                        progress,
                        f"Analisando tema '{new_topic}'... ({idx + 1}/{total_sentences})"
                    )
                    
                    # Check if the sentence contains the new topic
                    is_sensitive, similar_words = self.find_sensitive_content(
                        sentence_data["text"],
                        new_topic
                    )
                    if is_sensitive:
                        sentence_data["themes"][new_topic] = similar_words
                
                # Update progress and close window
                progress_window.update_progress(1.0, "Análise concluída!")
                self.root.after(1000, progress_window.destroy)
                
                # Create checkbox for the new topic
                checkbox = ctk.CTkCheckBox(
                    self.topics_frame,
                    text=new_topic,
                    variable=self.topics[new_topic],
                    command=self.filter_transcription,
                    fg_color=color,
                    text_color="white"
                )
                checkbox.pack(anchor=tk.W, padx=5, pady=2)
                
                # Update the display
                self.filter_transcription()
                
            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Erro", f"Erro ao analisar o novo tema: {str(e)}")
                
                # Remove the topic if analysis failed
                if new_topic in self.topics:
                    del self.topics[new_topic]
                if new_topic in self.topic_related_words:
                    del self.topic_related_words[new_topic]
                if new_topic in self.topic_colors:
                    del self.topic_colors[new_topic]
    
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

    def create_sentence_frame(self, text, start_time, end_time, topics_found=None):
        """Cria um frame para uma sentença com botão de reprodução e texto editável"""
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
        
        # Frame para o texto com fundo preto
        text_frame = ctk.CTkFrame(frame, fg_color="black")
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Texto editável da transcrição
        text_entry = ctk.CTkTextbox(
            text_frame,
            fg_color="black",
            text_color="white",
            height=60,
            wrap=tk.WORD
        )
        text_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_entry.insert("1.0", f"[{start_time:.2f}s - {end_time:.2f}s] {text}")
        
        if topics_found:
            # Cria labels para mostrar os temas encontrados
            topics_frame = ctk.CTkFrame(frame, fg_color="transparent")
            topics_frame.pack(side=tk.RIGHT, padx=5)
            
            for topic, words in topics_found.items():
                topic_label = ctk.CTkLabel(
                    topics_frame,
                    text=topic,
                    fg_color=self.topic_colors[topic],
                    text_color="white",
                    corner_radius=6,
                    padx=6,
                    pady=2
                )
                topic_label.pack(side=tk.LEFT, padx=2)
        
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
            # Clear previous transcription
            for widget in self.transcription_text.winfo_children():
                widget.destroy()
            self.matches_text.delete("1.0", tk.END)

            # Create progress window
            self.progress_window = ProgressWindow(self.root)
            self.progress_window.update_progress(0, "Carregando modelo de transcrição...")
            
            # Load and process audio file
            result = self.model.transcribe(self.filename)
            
            # Update progress window
            self.progress_window.update_progress(0.3, "Transcrevendo áudio...")
            
            # Store sentences and initialize processed_sentences
            self.sentences = [(segment["text"].strip(), segment["start"], segment["end"]) 
                            for segment in result["segments"]]
            
            total_sentences = len(self.sentences)
            self.processed_sentences = []
            
            # Pre-process all sentences for themes
            self.progress_window.update_progress(0.4, "Analisando temas sensíveis...")
            
            for idx, (text, start, end) in enumerate(self.sentences):
                # Update progress
                progress = 0.4 + (0.5 * (idx / total_sentences))
                self.progress_window.update_progress(
                    progress,
                    f"Analisando temas sensíveis... ({idx + 1}/{total_sentences})"
                )
                
                # Process each sentence
                sentence_data = {
                    "text": text,
                    "start": start,
                    "end": end,
                    "themes": {}
                }
                
                # Check for all possible themes
                for topic in self.topics.keys():
                    if topic != "Nenhum":
                        is_sensitive, similar_words = self.find_sensitive_content(text, topic)
                        if is_sensitive:
                            sentence_data["themes"][topic] = similar_words
                
                self.processed_sentences.append(sentence_data)
            
            # Display initial transcription
            self.progress_window.update_progress(0.9, "Montando interface...")
            self.filter_transcription()
            
            # Enable PDF button
            self.pdf_button.configure(state="normal")
            
            # Update progress and close window
            self.progress_window.update_progress(1.0, "Processamento concluído!")
            self.root.after(1000, self.progress_window.destroy)  # Fecha após 1 segundo
            
        except Exception as e:
            if hasattr(self, 'progress_window'):
                self.progress_window.destroy()
            messagebox.showerror("Erro", f"Erro ao processar o áudio: {str(e)}")

    def filter_transcription(self, selected_topic=None):
        """Filtra a transcrição para mostrar apenas o tema selecionado"""
        # Limpa a área de transcrição
        for widget in self.transcription_text.winfo_children():
            widget.destroy()
            
        if not hasattr(self, 'processed_sentences'):
            return
            
        # Para cada sentença processada
        for sentence_data in self.processed_sentences:
            has_any_topic = False
            active_themes = {}
            
            # Verifica temas ativos
            for topic, var in self.topics.items():
                if topic != "Nenhum" and var.get():
                    if topic in sentence_data["themes"]:
                        active_themes[topic] = sentence_data["themes"][topic]
                        has_any_topic = True
            
            # Mostra a sentença se:
            # - Tem algum tema detectado, OU
            # - "Nenhum" está marcado e não tem nenhum tema detectado
            if has_any_topic or (self.topics["Nenhum"].get() and not has_any_topic):
                self.create_sentence_frame(
                    sentence_data["text"],
                    sentence_data["start"],
                    sentence_data["end"],
                    active_themes if active_themes else None
                )

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
