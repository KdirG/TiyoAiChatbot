# knowledge_base_manager.py
import os
import torch
import numpy as np
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SIMILARITY_THRESHOLD, TOP_K_CHUNKS

def load_documents(directory):
    docs = []
    if not os.path.exists(directory):
        print(f"Uyarı: Bilgi tabanı dizini bulunamadı: {directory}")
        return []
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                docs.append({"filename": fname, "content": f.read()})
    print(f"{len(docs)} belge yüklendi.")
    return docs

def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def prepare_knowledge_base(directory, embedding_model, device):
    print("Bilgi tabanı hazırlanıyor ve embedding'ler oluşturuluyor...")
    docs = load_documents(directory)
    chunks = []
    for doc in docs:
        for ch in chunk_text(doc["content"]):
            chunks.append({"text": ch, "filename": doc["filename"]})
    
    if chunks:
        texts = [ch["text"] for ch in chunks]
        # Embedding modelini doğrudan fonksiyona geçir
        embeds = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True).to(device)
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeds[i]
        print(f"{len(chunks)} parça oluşturuldu ve embedding'leri hesaplandı.")
    else:
        print("Bilgi tabanında hiç parça bulunamadı.")
    return chunks

def retrieve_relevant_chunks(query_embedding, knowledge_base, top_k=TOP_K_CHUNKS, similarity_threshold=SIMILARITY_THRESHOLD):
    if not knowledge_base:
        return []
    all_embeds = torch.stack([chunk["embedding"] for chunk in knowledge_base])
    sims = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), all_embeds)
    top_k_scores, top_k_indices = torch.topk(sims, min(top_k, len(sims)))
    
    retrieved_texts = []
    for i, idx in enumerate(top_k_indices):
        if top_k_scores[i] >= similarity_threshold:
            retrieved_texts.append(knowledge_base[idx]["text"])
        else:
            # Eşik altındaki parçaları almayın
            break 
    print(f"{len(retrieved_texts)} alakalı parça getirildi (eşik {similarity_threshold}).")
    return retrieved_texts