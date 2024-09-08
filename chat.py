from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import google.generativeai as dalle  # Import for DALL-E-like image generation (adjust as needed)
import pymupdf  # PyMuPDF
import textwrap
import numpy as np
import pandas as pd
import random
from pathlib import Path

# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

app = Flask(__name__)

class Gemini:
    # Configuration
    pdf_path = "article"
    EMBEDDING_MODEL = 'models/text-embedding-004'
    TEXT_MODEL = genai.GenerativeModel('gemini-1.5-flash')
    API_KEY = 'AIzaSyB46Vu591CrCnF45sESz0wWJaAvNR1B0Dg'

    # Local storage for conversation history and embeddings (instead of MongoDB)
    conversation_history = []
    embeddings = {}  # Dictionary to store embeddings
    all_texts = {}
    df = pd.DataFrame()

    def __init__(self):
        genai.configure(api_key=self.API_KEY)
        self.df = self.createEmbedDataFrame(self.pdf_path)
        print("Gemini Initialized")

    # Extract text from PDFs
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        print("PDFs OCR'd")
        return text

    # Split text into chunks for embedding
    def split_text_into_chunks(self, catalog, max_chunk_size=9000):
        chunks = []
        for entry in catalog:
            words = catalog[entry].split()

            current_chunk = []
            current_size = 0

            for word in words:
                word_size = len(word) + 1  # Adding 1 for the space
                if current_size + word_size > max_chunk_size:
                    chunks.append({"Title": entry, "Text": " ".join(current_chunk)})
                    current_chunk = [word]
                    current_size = word_size
                else:
                    current_chunk.append(word)
                    current_size += word_size

            if current_chunk:
                chunks.append({"Title": entry, "Text": " ".join(current_chunk)})
        print("Texts split")
        return chunks

    def createEmbedDataFrame(self, pdf_directory):
        directory = Path.cwd() / pdf_directory
        for pdf_file in directory.glob('*.pdf'):
            self.all_texts[pdf_file.name] = self.extract_text_from_pdf(pdf_file)
        documents = self.split_text_into_chunks(self.all_texts)
        dataf = pd.DataFrame(documents)
        print("Dataframe created")
        dataf.columns = ['Title', 'Text']
        dataf['Embeddings'] = dataf.apply(lambda row: self.get_or_generate_embedding(row['Title'], row['Text']), axis=1)
        print("Embeddings created")
        return dataf

    def get_or_generate_embedding(self, title, text):
        embedding = self.embeddings.get((title, text))
        if embedding is None:
            embedding = self.embed_fn(title, text)
            self.embeddings[(title, text)] = embedding
        return embedding

    def embed_fn(self, title, text):
        return genai.embed_content(
            model=self.EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            title=title
        )["embedding"]

    def find_best_passage(self, query):
        query_embedding = genai.embed_content(
            model=self.EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        dot_products = np.dot(np.stack(self.df['Embeddings']), query_embedding["embedding"])
        idx = np.argmax(dot_products)
        return self.df.iloc[idx]['Text']

    # Format query with a prompt
    def makeQuery(self, quest):
        query = quest
        passage = self.find_best_passage(query)
        if query.strip() == '':
            return False
        try:
            escaped = passage.replace("'", "").replace('"', "").replace("\n", " ")
            full_message = textwrap.dedent(f"""
                You are role-playing as an oncology advice nurse talking to the patient or caregiver, both answering their questions and asking relevant questions to get more information. 
                Focus only on answering based on the knowledge base regarding colon cancer and colorectal cancer.
                The patient has never seen you before. You can suggest questions and respond based on the reference passage.
                MESSAGE: '{query}'
                PASSAGE: '{passage}'
                ANSWER:
            """)
            return full_message
        except Exception as e:
            print(f"GEMINI: {e}")
            return "Passage did not create"


# Initialize Gemini
akinator = Gemini()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message')
    if user_message:
        full_message = akinator.makeQuery(user_message)
        if full_message:
            chat = akinator.TEXT_MODEL.start_chat(history=[])
            response = chat.send_message(full_message)
            answer = response.text

            # Image generation based on the response text
            image_prompt = f"Generate an image related to this response: {answer}"  # Adjust prompt as needed
            try:
                image_data = dalle.text2im(
                    prompt=image_prompt,
                    size="1024x1024"  # Image size
                )
                image_url = image_data['image_url']  # Adjust depending on API response
            except Exception as e:
                print(f"Error generating image: {e}")
                image_url = None

            # Store the conversation and image locally in conversation_history list
            akinator.conversation_history.append({
                'user_message': user_message,
                'bot_response': answer,
                'image_url': image_url  # Save image URL
            })
            print("Data saved locally with image.")

            return jsonify({'response': answer, 'image_url': image_url})

    return jsonify({'response': "Invalid input"})

if __name__ == '__main__':
    app.run(debug=True)
