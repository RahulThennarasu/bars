from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient
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
    # All of this should come from config.py
    pdf_path = "article"
    EMBEDDING_MODEL = 'models/text-embedding-004'
    TEXT_MODEL = genai.GenerativeModel('gemini-1.5-flash')
    API_KEY = 'AIzaSyB7_quPu4GIRIsaNTG5D9XMMgjXzB4m7a8'

    # MongoDB connection details
    MONGO_URI = 'mongodb+srv://rahulthennarasu07:lego3011@cluster.igpxjoe.mongodb.net/?retryWrites=true&w=majority&appName=cluster'
    client = MongoClient(MONGO_URI)
    db = client['chat_data']
    collection = db['conversations']
    embedding_collection = db['embeddings']

    # Some dataframes/lists to store passages
    all_texts = {}
    df = pd.DataFrame()

    def __init__(self):
        genai.configure(api_key=self.API_KEY)
        self.df = self.createEmbedDataFrame(self.pdf_path)
        print("Gemini Initialized")

    # Extract files from PDFs
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        print("PDFs OCR'd")
        return text

    # Split text into chunks so that it is embeddable (more than 10000 bytes doesn't get embedded)
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
        print(Path.cwd())
        directory = Path.cwd() / pdf_directory
        print(directory)
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
        embedding = self.retrieve_embedding_from_db(title, text)
        if embedding is None:
            embedding = self.embed_fn(title, text)
            self.save_embedding_to_db(title, text, embedding)
        return embedding

    def embed_fn(self, title, text):
        return genai.embed_content(
            model=self.EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            title=title
        )["embedding"]

    def save_embedding_to_db(self, title, text, embedding):
        self.embedding_collection.insert_one({
            'title': title,
            'text': text,
            'embedding': embedding
        })

    def retrieve_embedding_from_db(self, title, text):
        result = self.embedding_collection.find_one({'title': title, 'text': text})
        return result['embedding'] if result else None

    def find_best_passage(self, query):
        query_embedding = genai.embed_content(
            model=self.EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        dot_products = np.dot(np.stack(self.df['Embeddings']), query_embedding["embedding"])
        idx = np.argmax(dot_products)
        return self.df.iloc[idx]['Text']

    # Format query with a prompt, so that it answers the way we intend using the embeddings
    def makeQuery(self, quest):
        query = quest
        passage = self.find_best_passage(query)
        if query.strip() == '':
            return False
        try:
            escaped = passage.replace("'", "").replace('"', "").replace("\n", " ")
            full_message = textwrap.dedent(f"""
                You are role playing as a oncology advise nurse talking to the patient or a caregiver, both answering their questions, but also asking \
    relevant questions to get more information. Once they tell you about the diagnosis and treatment focus only on answering the questions and \
    giving useful information. Currently your knowledge is limited to colon cancer and colorectal cancer. Patient has never seen you before. You can suggest questions to the patient and answer to your ability based on the the knowledge base.\
    the tests. Also do not repeat questions multiple times. Please be sure to respond to the user's questions using the text from the reference passage included below. \
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

            # Save the conversation to MongoDB
            try:
                akinator.collection.insert_one({
                    'user_message': user_message,
                    'bot_response': answer
                })
                print("Data saved to MongoDB.")
            except Exception as db_e:
                print(f"Failed to save to MongoDB: {db_e}")

            return jsonify({'response': answer})

    return jsonify({'response': "Invalid input"})


if __name__ == '__main__':
    app.run(debug=True)
