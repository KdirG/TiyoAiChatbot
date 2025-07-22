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
    if not user_input_tr.strip():
        return "Lütfen bir şey yazın.", history

    input_en = translate_tk_to_en(user_input_tr, translator_tk_en_tokenizer, translator_tk_en_model, device)
    print(f"\n[DEBUG] Opus-MT T-EN Çevirisi: '{user_input_tr}' -> '{input_en}'")
    
    # KULLANICININ MEVCUT SORUSUNU GEÇMİŞE EKLE (LLM'e gönderilmeden önce)
    # Bu, LLM'in kendi içsel bağlamında bu mesajı görmesini sağlar.
    history.append({"role": "user", "content": input_en})

    # RAG için kullanılacak sorguyu oluşturma (Bağlamsallaştırma)
    contextualized_query_en = input_en
    # Eğer geçmişte en az bir önceki AI yanıtı varsa (yani history'de en az 2 mesaj var ve sonuncusu kullanıcı sorusu)
    # ve mevcut kullanıcı girdisi kısa/genel bir takip sorusu niteliğindeyse,
    # önceki yapay zeka yanıtı ile birleştirerek sorguyu zenginleştir.
    if len(history) >= 2 and \
       len(input_en.split()) < 5 and \
       any(phrase in input_en.lower() for phrase in ["explain", "more", "why", "how", "what do you mean", "açıkla", "neden", "nasıl", "ne demek istiyorsun"]):
        
        # history[-1] şu anki kullanıcının sorusu, history[-2] ise AI'ın son cevabı.
        last_assistant_response_en = history[-2]['content'] 
        contextualized_query_en = last_assistant_response_en + ". " + input_en
        print(f"[DEBUG] Bağlamsallaştırılmış RAG Sorgusu: '{contextualized_query_en}'")
    
    # RAG sorgusu için gömü oluşturma (bağlamsallaştırılmış sorguyu kullanıyoruz)
    if contextualized_query_en in question_embedding_cache:
        query_embedding = question_embedding_cache[contextualized_query_en]
    else:
        query_embedding = embedding_model.encode(contextualized_query_en, convert_to_tensor=True).to(device)
        question_embedding_cache[contextualized_query_en] = query_embedding

    retrieved_chunks = retrieve_relevant_chunks(query_embedding, kb_data)
    context_text = "\n".join(retrieved_chunks)
    
    # Eğer RAG bağlamı bulunamazsa, LLM'e göndermeden doğrudan yanıt ver
    if not retrieved_chunks:
        # history'ye "bağlam yok" cevabını ekle (LLM'i bypass ettiğimiz için bu cevabı kendimiz ekliyoruz)
        history.append({"role": "assistant", "content": "Üzgünüm, ancak bağlamda bu soruya ait bilgi yok."})
        return "Üzgünüm, Sadece TİYO ile ilgili soruları cevaplayabilirim.", history

    # LLM'e gönderirken tüm güncel geçmişi kullanıyoruz (artık içinde kullanıcının son mesajı da var)
    llm_response = send_message_to_llm(input_en, history, context_text, LM_STUDIO_URL)
    print(f"[DEBUG] Dil Modelinden Gelen İngilizce Yanıt: '{llm_response}'")
    
    llm_response = apply_output_filter(llm_response)
    
    # AI'ın yanıtını geçmişe ekle
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
        # opus_llm_agent_step fonksiyonu history'yi kendi içinde güncelliyor ve yeni history'yi döndürüyor
        # Bu yüzden burada history'yi tekrar atamak önemli.
        response, history = opus_llm_agent_step(user_input, history, kb)
        print(f"AI (Türkçe): {response}")