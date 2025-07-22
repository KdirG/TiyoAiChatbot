# models.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch

def load_translation_models(device):
    print("Çeviri modelleri yükleniyor...")
    translator_tk_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-tr-en")
    translator_tk_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-tr-en").to(device)
    
    translator_en_tk_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")
    translator_en_tk_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr").to(device)
    print("Çeviri modelleri yüklendi.")
    return translator_tk_en_model, translator_tk_en_tokenizer, translator_en_tk_model, translator_en_tk_tokenizer

def load_embedding_model(device):
    print("Embedding modeli yükleniyor...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    print("Embedding modeli yüklendi.")
    return embedding_model