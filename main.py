# main.py
import os
import nltk
import torch

# Diğer modüllerden gerekli fonksiyonları/sınıfları içe aktar
from config import KNOWLEDGE_BASE_DIR, LM_STUDIO_URL
from models import load_translation_models, load_embedding_model
from knowledge_base_manager import prepare_knowledge_base, retrieve_relevant_chunks
from llm_interface import send_message_to_llm
from utils import translate_tk_to_en, translate_en_to_tk_segmented, apply_output_filter

# NLTK punkt modelini kontrol etme/indirme
print("NLTK 'punkt' modelini kontrol ediyor/indiriyor...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Cihazı belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Modelleri yükle
translator_tk_en_model, translator_tk_en_tokenizer, translator_en_tk_model, translator_en_tk_tokenizer = load_translation_models(device)
embedding_model = load_embedding_model(device)

question_embedding_cache = {} # Önbellek burada kalabilir veya ayrı bir cache modülüne taşınabilir

def opus_llm_agent_step(user_input_tr: str, history: list, kb_data: list):
    # Fonksiyonun geri kalan mantığı burada olacak, ancak içindeki çağrılar
    # artık ilgili modüllerden gelecek.

    if not user_input_tr.strip():
        return "Lütfen bir şey yazın.", history

    input_en = translate_tk_to_en(user_input_tr, translator_tk_en_tokenizer, translator_tk_en_model, device)
    print(f"\n[DEBUG] Opus-MT T-EN Çevirisi: '{user_input_tr}' -> '{input_en}'")
    
    if input_en in question_embedding_cache:
        query_embedding = question_embedding_cache[input_en]
    else:
        query_embedding = embedding_model.encode(input_en, convert_to_tensor=True).to(device)
        question_embedding_cache[input_en] = query_embedding

    retrieved_chunks = retrieve_relevant_chunks(query_embedding, kb_data)
    context_text = "\n".join(retrieved_chunks)
    
    if not retrieved_chunks:
        history.append({"role": "user", "content": input_en})
        history.append({"role": "assistant", "content": "Üzgünüm, ancak bağlamda bu soruya ait bilgi yok."})
        return "Üzgünüm, Sadece TİYO ile ilgili soruları cevaplayabilirim.", history

    current_chat_for_llm = [{"role": "user", "content": input_en}]

    llm_response = send_message_to_llm(input_en, current_chat_for_llm, context_text, LM_STUDIO_URL)
    print(f"[DEBUG] Dil Modelinden Gelen İngilizce Yanıt: '{llm_response}'")
    
    llm_response = apply_output_filter(llm_response)
    history.append({"role": "assistant", "content": llm_response})

    return translate_en_to_tk_segmented(llm_response, translator_en_tk_tokenizer, translator_en_tk_model, device), history


if __name__ == "__main__":
    kb = prepare_knowledge_base(KNOWLEDGE_BASE_DIR, embedding_model, device)
    history = []

    print("\n--- AI Sohbet Başlatılıyor --- (Çıkmak için 'çıkış' yazın.)")
    while True:
        user_input = input("\nSen (Türkçe): ")
        if user_input.lower() == "çıkış":
            break
        response, history = opus_llm_agent_step(user_input, history, kb)
        print(f"AI (Türkçe): {response}")