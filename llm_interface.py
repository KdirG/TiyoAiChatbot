# llm_interface.py
import requests
import json
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS

def send_message_to_llm(prompt, history, retrieved_context="", url="http://192.168.1.25:1234/v1/chat/completions"):
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use only provided context. Do not use general knowledge.Tiyo is a name, do not try to translate it."}
    ]
    if retrieved_context:
        messages.append({"role": "system", "content": f"Relevant info:\n{retrieved_context}"})
    
    # History (kısa olan) buraya eklenecek
    messages += history 
    messages.append({"role": "user", "content": prompt}) # En son kullanıcı sorusu

    try:
        res = requests.post(url, headers=headers, json={"messages": messages, "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS})
        res.raise_for_status() # HTTP hatalarını yakala
        return res.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API isteği hatası: {e}")
        return f"[API Bağlantı Hatası] {str(e)}"
    except KeyError:
        print(f"API yanıtı beklenenden farklı: {res.json()}")
        return "[API Yanıt Hatası] Beklenmedik format."
    except Exception as e:
        return f"[Genel API Hatası] {str(e)}"