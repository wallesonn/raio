# Raio Project - Review any Audio with Intelligent Outputs
Projeto de TCC para conclusao da Pos-Graduação em Inteligência Artificial da UNDB São Luís/MA
Autores: Walleson Ferreira e Jandy Cidreira

# Projeto de Processamento de Áudio com IA

Este projeto utiliza inteligência artificial para processar arquivos de áudio e oferece as seguintes funcionalidades:
	1.	Transcrição do áudio com identificação de cada falante.
	2.	Detecção de temas sensíveis na transcrição, como drogas, morte e crimes sexuais.
	3.	Seleção de temas a serem identificados pelo usuário.
	4.	Visualização do áudio e dos resultados em uma interface gráfica intuitiva.

# Tecnologias Utilizadas
	•	OpenAI Whisper: Para transcrição de áudio com alta precisão.
	•	Spacy e NLTK: Para análise de linguagem natural e identificação de temas.
	•	Librosa e Matplotlib: Para manipulação e visualização do áudio.
	•	TKinter Modern Theme: Para criar a interface gráfica.

# Como Utilizar

Pré-requisitos
	1.	Instale o Python 3.7+ em seu ambiente.
	2.	Certifique-se de ter as bibliotecas necessárias instaladas:

pip install whisper flask librosa matplotlib nltk

# Passo a Passo

1. Clone o Repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

2. Instale as dependências:

pip install -r requirements.txt

3. Execute o Servidor Flask:

python app.py

4. Acesse a Interface:

Abra o navegador e acesse http://127.0.0.1:5000/.

# Funcionalidades
	1.	Transcrição do Áudio:
O projeto utiliza o modelo OpenAI Whisper para transcrever áudios com alta precisão e identificar os falantes.
	2.	Identificação de Temas Sensíveis:
O usuário pode selecionar os temas desejados (drogas, morte, ou crimes sexuais) para verificar em quais partes da transcrição esses assuntos aparecem.
	3.	Visualização do Áudio:
A interface gráfica exibe a forma de onda (waveform) do áudio processado para facilitar a análise visual.

# Estrutura do Projeto

├── app.py               # Código principal com lógica do Flask e processamento
├── templates/
│   └── index.html       # Template HTML para a interface
├── static/
│   └── audio_waveform.png  # Arquivo gerado para visualização do áudio
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação do projeto

# Exemplo de Uso
	1.	Faça o upload de um arquivo de áudio.
	2.	Selecione os temas que deseja analisar.
	3.	Visualize a transcrição, os temas detectados e a forma de onda do áudio na interface gráfica.

Visualização do Áudio

# Contribuindo

Contribuições são bem-vindas!
Sinta-se à vontade para abrir Issues ou enviar Pull Requests com melhorias ou correções.

# Licença

Este projeto é licenciado sob a MIT License.
