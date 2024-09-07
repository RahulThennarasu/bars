import google.generativeai as genai
from pymongo import MongoClient
import pymupdf # PyMuPDF
import textwrap
import numpy as np
import pandas as pd
import random
from pathlib import Path
# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

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

    #some dataframes/lists to store passages
    all_texts = {}
    df = pd.DataFrame()

    #on creation configures and makes necessary dataframe
    def __init__(self):
        genai.configure(api_key=self.API_KEY)
        self.df = self.createEmbedDataFrame(self.pdf_path)
        # Consider using logging.py instead of print.
        print("started")

    #extracts files from pdfs
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        print("pdfs ocrd")
        return text

    #splits text into chunks so that it is embeddable(more than 10000 bytes doesn't get embedded)
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
        print("texts split")
        return chunks

    def createEmbedDataFrame(self, pdf_directory):
        print(Path.cwd())
        directory = Path.cwd() / pdf_directory
        print(directory)
        for pdf_file in directory.glob('*.pdf'):
            self.all_texts[pdf_file.name] = self.extract_text_from_pdf(pdf_file)
        documents = self.split_text_into_chunks(self.all_texts)
        dataf = pd.DataFrame(documents)
        print("dataframe made")
        dataf.columns = ['Title', 'Text']
        dataf['Embeddings'] = dataf.apply(lambda row: self.get_or_generate_embedding(row['Title'], row['Text']), axis=1)
        print("embeddings made")
        return dataf

    def get_or_generate_embedding(self, title, text):
        #
        embedding = self.retrieve_embedding_from_db(title, text)
        if embedding is None:
            embedding = self.embed_fn(title, text)
            self.save_embedding_to_db(title, text, embedding)
        return embedding

    def embed_fn(self, title, text):
        #makes embeds from the dataframe
        return genai.embed_content(model=self.EMBEDDING_MODEL,
                                   content=text,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

    def save_embedding_to_db(self, title, text, embedding):
        self.embedding_collection.insert_one({
            'title': title,
            'text': text,
            'embedding': embedding
        })

    def retrieve_embedding_from_db(self, title, text):
        result = self.embedding_collection.find_one({
            'title': title,
            'text': text
        })
        return result['embedding'] if result else None

    def find_best_passage(self, query):
        #compares passage database embeds to query embeds to find most relevant passage
        relevant_dataframe = self.df
        query_embedding = genai.embed_content(model=self.EMBEDDING_MODEL,
                                              content=query,
                                              task_type="retrieval_query")
        dot_products = np.dot(np.stack(relevant_dataframe['Embeddings']), query_embedding["embedding"])
        idx = np.argmax(dot_products)
        return relevant_dataframe.iloc[idx]['Text']

    #format query with a prompt, so that it answers the way we intend(using the embeds)
    def makeQuery(self, quest):
        query = quest
        #finds the most relevant section of the passage to list in the geminiQuery as context
        passage = self.find_best_passage(query)
        #ensures that there is an actual query
        if query.strip() == '':
            return False
        try:
            escaped = passage.replace("'", "").replace('"', "").replace("\n", " ")
            # To iterate quickly on this, I would make this configurable, probably by reading a row from the db. That way we can iterate on various prompt designs without needing to rebuild.
            full_message = textwrap.dedent(f"""You are role playing as a oncology advise nurse talking to the patient or a caregiver, both answering their questions, but also asking \
    relevant questions to get more information. Once they tell you about the diagnosis and treatment focus only on answering the questions and \
    giving useful information. Currently your knowledge is limited to colon cancer and colorectal cancer. Patient has never seen you before. You can suggest questions to the patient and answer to your ability based on the the knowledge base.\
    the tests. Also do not repeat questions multiple times. Please be sure to respond to the user's questions using the text from the reference passage included below. \
    MESSAGE: '{query}'
    PASSAGE: '{passage}'
  
      ANSWER:""").format(query=query, passage=escaped)
 #            full_message = textwrap.dedent(f"""You are a helpful and informative bot that responds and answers messages using text from the reference passage included below. \
 #  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
 #  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
 #  strike a friendly and converstional tone. \
 #  If possible utilize information from the passage, but if it is irrelevant you may ignore it. Your priority is delivering
 # an accurate and relevant response to the message NOT THE PASSAGE. ENSURE THAT YOUR RESPONSE IS RELEVANT TO THE MESSAGE IRRELEVANT OF THE PASSAGE.
 #  MESSAGE: '{query}'
 #  PASSAGE: '{passage}'
 #
 #    ANSWER:
 #    """).format(query=query, passage=escaped)
            return full_message
        except Exception as e:
            print(f"GEMINI: {e}")
            # Better error message is needed.
            return "Passage did not create"

#practice convo with gemini by running this file
if __name__ == '__main__':
    Akinator = Gemini()
    chat = Akinator.TEXT_MODEL.start_chat(history=[])
    while True:
        question = input("You: ")
        full_message = Gemini.makeQuery(Akinator, question)
        if full_message == False:
            break
        print("GEMINI PINGED")
        response = chat.send_message(full_message)
        answer = response.text
        print('\n')
        print(f"Bot: {response.text}")
        print('\n')
        try:
            Akinator.collection.insert_one({
                'user_message': question,
                'bot_response': response.text
            })
            print("Data saved to MongoDB.")
        except Exception as db_e:
            print(f"Failed to save to MongoDB: {db_e}")
