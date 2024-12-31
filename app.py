import os
# Configurar backend não-interativo antes de importar pyplot
import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo

from flask import Flask, request, render_template, jsonify
import whisper
import librosa
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    """Download todos os recursos NLTK necessários."""
    try:
        # Recursos necessários
        resources = [
            'punkt',
            'punkt_tab',
            'stopwords',
            'averaged_perceptron_tagger'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Erro ao baixar recurso {resource}: {str(e)}")
                
    except Exception as e:
        print(f"Erro ao baixar recursos NLTK: {str(e)}")

# Download required NLTK data
download_nltk_resources()

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Whisper model
model = whisper.load_model("base")

# Define sensitive topics
SENSITIVE_TOPICS = {
	'drogas': ['droga', 'cocaina', 'maconha', 'crack', 'tráfico'],
	'morte': ['morte', 'assassinato', 'homicídio', 'matar', 'morrer'],
	'crimes_sexuais': ['estupro', 'abuso', 'violência sexual', 'assédio'],
	'familia': ['irmão', 'irmã', 'pai', 'mãe', 'filho', 'filha', 'família','irmãos', 'irmãs', 'pais', 'filhos', 'filhas'],
}

def analyze_audio(audio_path, selected_topics):
	# Transcribe audio using Whisper
	result = model.transcribe(audio_path)
	transcription = result['text']
	
	# Generate audio visualization
	y, sr = librosa.load(audio_path)
	plt.figure(figsize=(10, 4))
	plt.plot(y)
	plt.title('Audio Waveform')
	plt.xlabel('Time')
	plt.ylabel('Amplitude')
	waveform_path = 'static/audio_waveform.png'
	plt.savefig(waveform_path)
	plt.close()
	
	# Analyze topics
	tokens = word_tokenize(transcription.lower())
	found_topics = {}
	
	for topic, keywords in SENSITIVE_TOPICS.items():
		if topic in selected_topics:
			found_topics[topic] = []
			for i, token in enumerate(tokens):
				if token in keywords:
					context = ' '.join(tokens[max(0, i-5):min(len(tokens), i+6)])
					found_topics[topic].append(context)
	
	return {
		'transcription': transcription,
		'topics': found_topics,
		'waveform': waveform_path
	}

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		if 'audio' not in request.files:
			return jsonify({'error': 'No audio file uploaded'})
		
		audio_file = request.files['audio']
		if audio_file.filename == '':
			return jsonify({'error': 'No selected file'})
		
		# Get selected topics
		selected_topics = request.form.getlist('topics')
		
		# Save and process audio file
		audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
		audio_file.save(audio_path)
		
		# Analyze audio
		results = analyze_audio(audio_path, selected_topics)
		
		return render_template('index.html', results=results)
	
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
