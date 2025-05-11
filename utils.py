# utils.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import streamlit as st
import spacy
from collections import Counter, defaultdict
from transformers import pipeline
import openai
from openai import OpenAI
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load embedding model
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()



# Chunking text
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# Create FAISS index dynamically
def process_text(text):
    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings_np = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return chunks, index

# Retrieve top-k chunks from FAISS
def get_top_chunks(query, index, chunks, k=5):
    query_vector = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]



# Load FLAN-T5 large a generative model understanding the context and giving human responses (use "google/flan-t5-large" when good ram)
@st.cache_resource(show_spinner="Loading FLAN-T5 model...")
def load_flan_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer_flan, model_flan = load_flan_model()
# Generate answer with FLAN-T5
def generate_answer_with_rag(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    prompt = (
        f"Based on the following context from a novel, answer the question clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:")

    inputs = tokenizer_flan(prompt, return_tensors="pt", max_length=1024, truncation=True)

    outputs = model_flan.generate(
        inputs["input_ids"],
        max_length=350,
        min_length=50,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer_flan.decode(outputs[0], skip_special_tokens=True)


# Extract simple ontology (top N frequent entities)
@st.cache_data(show_spinner="🔍 Extracting top ontology...")
def extract_ontology(text, top_n=10):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Map short-form labels to full-form names
    label_map = {
        "PERSON": "People",
        "GPE": "Geopolitical Locations",
        "ORG": "Organizations"
    }

    entity_counter = defaultdict(Counter)
    for ent in doc.ents:
        if ent.label_ in label_map:
            entity_counter[ent.label_][ent.text.strip()] += 1

    ontology = {
        label_map[label]: [entity for entity, _ in counter.most_common(top_n)]
        for label, counter in entity_counter.items()
    }
    return ontology



#Extra gen ai model below

#4444444444444GPT- Quota Expired
#client = OpenAI(api_key="<---paste your api key here--->")

#using GPT
#def generate_answer_with_rag(query, retrieved_chunks):
#    context = " ".join(retrieved_chunks)
#
#    prompt = (
#        f"You are a helpful assistant answering questions from a novel. "
#        f"Use only the context below to answer the question clearly and accurately.\n\n"
#        f"Context:\n{context}\n\n"
#        f"Question: {query}\n\n"
#        f"Answer:"
#    )
#
#    try:
#        response = client.chat.completions.create(
#            model="gpt-3.5-turbo",
#            messages=[{"role": "user", "content": prompt}],
#            temperature=0.7,
#            max_tokens=300,
#        )
#        return response.choices[0].message.content.strip()
#
#    except Exception as e:
#        return f"⚠️ GPT Error: {e}"



##22222222222model specifically for QA on contexts
#qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
## same model but handleing impossible answer
#qa_pipeline = pipeline("question-answering",model="deepset/roberta-base-squad2",tokenizer="deepset/roberta-base-squad2",handle_impossible_answer=True)
#def generate_answer_with_rag(query, retrieved_chunks):
#    context = " ".join(retrieved_chunks)
#    
#    if not context.strip():
#        #return "⚠️ No relevant context found."
#
#    try:
#        #result = qa_pipeline(question=query, context=context)
#        #return result.get("answer", "❓ Could not find a clear answer.")
#    except Exception as e:
#        return f"⚠️ Error during answer generation: {e}"



## 111111111111111Load T5 model and tokenizer A smaller basic model fast but not accurate
#model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
#tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
## Function to generate answer using RAG-like flow-- uncomment if using pretrained moel
#def generate_answer_with_rag(query, retrieved_chunks):
#    context = " ".join(retrieved_chunks)
#    #prompt = f"Question: {query}\nContext: {context}\nAnswer:"
#    prompt = (
#    f"Based on the following context from a novel, answer the question clearly.\n"
#    f"Context:\n{context}\n\n"
#    f"Question: {query}\nAnswer:")
#    inputs = tokenizer_t5(prompt, return_tensors="pt", max_length=512, truncation=True)
#    output = model_t5.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
#    return tokenizer_t5.decode(output[0], skip_special_tokens=True)