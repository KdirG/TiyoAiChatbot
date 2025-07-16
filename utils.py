# utils.py
import nltk

def translate_tk_to_en(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_en_to_tk(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_en_to_tk_segmented(english_text, tokenizer, model, device):
    sentences = nltk.tokenize.sent_tokenize(english_text, language='english')
    # Her cümleyi ayrı ayrı çevir
    translated_sentences = []
    for s in sentences:
        if s.strip(): # Boş cümleleri atla
            translated_sentences.append(translate_en_to_tk(s, tokenizer, model, device))
    return " ".join(translated_sentences)

def apply_output_filter(response_text):
    sentences = nltk.tokenize.sent_tokenize(response_text, language='english')
    cleaned = []
    for sentence in sentences:
        # İlk cümle her zaman alınır
        if not cleaned:
            cleaned.append(sentence.strip())
        # Soru işaretli cümleler veya sonrasında gelenleri kes
        elif "?" in sentence:
            break 
        else:
            cleaned.append(sentence.strip())
    
    result = " ".join(cleaned).strip()

    # Anahtar kelime filtreleri
    for kw in ["violence", "illegal", "dangerous"]: # Daha fazla kelime ekleyebilirsiniz
        if kw in result.lower():
            return "Üzgünüm, bu konuda bilgi sağlayamam." # Türkçe yanıt döndürün
    
    return result if result else "Üzgünüm, uygun bir yanıt bulunamadı." # Türkçe yanıt döndürün