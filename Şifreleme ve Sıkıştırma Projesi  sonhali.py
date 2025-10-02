import oqs
import os
from Crypto.Cipher import AES
import mimetypes
from datetime import datetime
import hashlib
import numpy as np
from scipy.stats import entropy as shannon_entropy
import tempfile
import time
import random
import torch
from PIL import Image
import clip  


# CLIP modeli yükle (eğitim gerektirmez) resim sıkıştırma için 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Görselden özellik çıkar
def extract_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        
        features = model.encode_image(image)
    return features / features.norm(dim=-1, keepdim=True)

# Adaptif kalite belirle (yaklaşım: içeriğin zenginliği)
def compute_quality(features):
    # Düz içeriğe yakın olan görsellere düşük kalite seçimi
    reference_texts = ["a blank image", "a very simple drawing", "highly detailed photo", "complex artwork"]
    text_tokens = clip.tokenize(reference_texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = features @ text_features.T  # cosine similarity uygulanır

    sim_scores = similarity.squeeze().cpu().numpy()

    # Basit benzerliğe göre kalite skalası (30–90 arası)
    blank_sim = sim_scores[0]
    detail_sim = sim_scores[2]
    quality = 30 + (detail_sim - blank_sim) * 60
    return int(max(30, min(90, quality)))

# Sıkıştır
def compress_adaptively_image(image_path):
    features = extract_features(image_path)
    quality = compute_quality(features)
    print(f"Belirlenen JPEG kalite: {quality}")
    out_path = image_path.replace(".", f"_compressed_{quality}.")
    img = Image.open(image_path)
    img.save(out_path, "JPEG", quality=quality)
    print(f"Sıkıştırılmış çıktı: {out_path}")

#Metin sıkıştırma
import re
import zipfile
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Stopwords genişletilmiş hali
turkce_stop_words = set(stopwords.words('turkish'))
ek_stopwords = {"bir", "ve", "de", "da", "ile", "ki", "bu", "şu", "o", "mu", "mı"}
turkce_stop_words = turkce_stop_words.union(ek_stopwords)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
#metni önişleme yapıyorum
def encode_text_and_create_zip(input_txt_path, output_zip_path, max_features=5):
    metin = read_text_file(input_txt_path)
    metin_islenmis = preprocess_text(metin)

    vectorizer = TfidfVectorizer(stop_words=list(turkce_stop_words), max_features=max_features)
    X = vectorizer.fit_transform([metin_islenmis])
    onemli_kelimeler = vectorizer.get_feature_names_out()

    token_map = {kelime: f"[KW{i}]" for i, kelime in enumerate(onemli_kelimeler)}

    def encode(text):
        text_lower = preprocess_text(text)
        for kelime, token in token_map.items():
            text_lower = re.sub(r'\b' + re.escape(kelime) + r'\b', token, text_lower)
        return text_lower

    encoded_text = encode(metin)

    token_to_word = {v: k for k, v in token_map.items()}
    token_map_json = json.dumps(token_to_word, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(output_zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.txt", encoded_text)
        zf.writestr("token_map.json", token_map_json)

    print(f"ZIP dosyası kaydedildi: {output_zip_path}")
#metin burada encode ediliyor ve zipleniyor zipin içine token map kaydediliyor
def decode_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, mode='r') as zf:
        encoded_text = zf.read("data.txt").decode('utf-8')
        token_map_loaded = json.loads(zf.read("token_map.json").decode('utf-8'))

        # token -> kelime mapini kelime->token mapine çeviriyoruz decode fonksiyonu için
        word_token_map_loaded = {v: k for k, v in token_map_loaded.items()}

        def decode(text, token_map):
            for kelime, token in token_map.items():
                text = text.replace(token, kelime)
            return text

        original_text = decode(encoded_text, word_token_map_loaded)
        return original_text

#Ses sıkıştırma 
import numpy as np

import torch
import soundfile as sf
import subprocess
from torchvggish import vggish, vggish_input


def extract_vggish_embedding(audio_path):
    #Model yükleme vggish modeli
    examples_batch = vggish_input.wavfile_to_examples(audio_path)
    model = vggish()
    model.eval()

    with torch.no_grad():
        # embedding çıkarımı 
        inputs = examples_batch
        embedding = model(inputs).numpy()

    embedding_mean = embedding.mean(axis=0)
    embedding_mean = np.atleast_1d(embedding_mean)
    norm = np.linalg.norm(embedding_mean)
    if norm > 0:
        embedding_mean /= norm
    return embedding_mean

#sesin süresini alma
def get_audio_duration(audio_path):
    with sf.SoundFile(audio_path) as f:
        duration = len(f) / f.samplerate
    return duration

#Çıkarılan özelliklerle bitrat değeri hesabı
def compute_bitrate_from_embedding(embedding, duration_sec):
    mean_val = np.mean(embedding)
    std_val = np.std(embedding)
    energy = np.linalg.norm(embedding)
    sparsity = np.sum(np.abs(embedding) < 0.1) / len(np.atleast_1d(embedding))

    score = (mean_val * 0.4) + (std_val * 0.3) + (energy * 0.2) + (sparsity * 0.1)

    print(score)
    if score < 0.2:
        bitrate = 24
    elif score < 0.35:
        bitrate = 48
    elif score < 0.5:
        bitrate = 96
    elif score < 0.7:
        bitrate = 128
    else:
        bitrate = 160

    if duration_sec < 3:
        bitrate = min(bitrate, 48)

    return bitrate

#Seçilen bitrat ile adaptif mp3 sıkıştırma
def compress_audio_ffmpeg(input_path, output_path, bitrate, duration_sec):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
    ]

    if duration_sec < 3:
        command += ["-ac", "1", "-ar", "16000"]  # mono ve 16 kHz

    command += [
        "-b:a", f"{bitrate}k",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✔️ Sıkıştırılmış dosya kaydedildi: {output_path} (Bitrate: {bitrate} kbps)")

########################################################################################
#Şifreleme kısmı
#Güncel şifreleme anahtarını alır Kyber versiyonunun
def get_current_kem_alg(path="kem_alg.txt"):
    if not os.path.exists(path):
        default_alg = "Kyber512"
        with open(path, "w") as f:
            f.write(default_alg)
        return default_alg

    with open(path, "r") as f:
        return f.read().strip()
#Şimdiki anahtar algoritmasını txtye kaydeder
def set_current_kem_alg(new_alg, path="kem_alg.txt"):
    with open(path, "w") as f:
        f.write(new_alg)
#Bir saldırı olduğunda otomatik olarak Anahtar seviyesi yükselir
def upgrade_kem_alg(current_alg, path="kem_alg.txt"):
    kem_alg_levels = ["Kyber512", "Kyber768", "Kyber1024"]
    try:
        current_index = kem_alg_levels.index(current_alg)
        if current_index < len(kem_alg_levels) - 1:
            new_alg = kem_alg_levels[current_index + 1]
            print(f"🔐 Kyber algoritması yükseltildi: {current_alg} → {new_alg}")
            set_current_kem_alg(new_alg, path)
            return new_alg
        else:
            print(f"✅ Zaten en yüksek güvenlik seviyesi: {current_alg}")
            return current_alg
    except ValueError:
        print(f"❌ Geçersiz Kyber algoritması: {current_alg}")
        return current_alg

def corrupt_bytes(data, num_bytes=5):
    data = bytearray(data)
    for _ in range(num_bytes):
        idx = random.randint(0, len(data) - 1)
        data[idx] ^= 0xFF
    return data
def clear_log_file(log_path="logs\operation_log.txt"):
    if os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("")  # Dosyayı sıfırla
        print("✅ Log dosyası temizlendi.")
    else:
        print("⚠️ Log dosyası bulunamadı.")

# === Yardımcı Fonksiyonlar ===
from datetime import datetime
def simulate_cpa_behavior_and_detect_attack(num_plaintexts=5):
    import tempfile
    #Burada CCA similasyonu yapıyoruz ilk başta anahtar üretimi ile başladık


    print("🔍 CPA Simülasyonu başlatılıyor...")

    try:
        with oqs.KeyEncapsulation(kem_alg) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            ciphertext, shared_secret_key = kem.encap_secret(public_key)
        #elimizde plaintetxtler var
        plaintexts = [
            "This is a test message.",
            "AAAAAAAAAAAAAAAAAAAAA",
            "1234567890" * 3,
            "Different plaintext for CPA test.",
            "Lorem ipsum dolor sit amet."
        ]
        #Burdaki plaintextleri txtye olarak kaydedip şifreliyoruz.
        for i in range(min(num_plaintexts, len(plaintexts))):
            text = plaintexts[i]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
                temp_file.write(text)
                temp_file_path = temp_file.name

            print(f"🧪 [{i+1}/{num_plaintexts}] Şifreleniyor: {temp_file_path}")
            aes_encrypt_file(temp_file_path, shared_secret_key)
            os.remove(temp_file_path)
        #Her aşamada işlemler loglanıyor
        print("\n📊 Özellikler çıkarılıyor...")
        features = extract_cumulative_features()
        for k, v in features.items():
            print(f"{k}: {v}")
        #her loglardan özellik çıkarımı yapıyoruz ve saldırı modellerine veriyoruz
        print("\n🧠 Saldırı tespiti yapılıyor...")
        result = defineattack(features) #Saldırıyı tespit edip sonucunu veriyoruz
        print(f"\n🎯 Sonuç: {result}")
        return result

    except Exception as e:
        print(f"❌ Simülasyon sırasında hata oluştu: {e}")
        return "Hata"


import os
import tempfile
import time
import random

def simulate_coa_behavior_and_detect_attack(num_ciphertexts=5):
    print("🔍 COA Simülasyonu başlatılıyor...")
    kem_alg = get_current_kem_alg()

    #COA similasyonu başaltıyoruz Burada 
    try:
        with oqs.KeyEncapsulation(kem_alg) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            _, shared_secret_key = kem.encap_secret(public_key)
        #Burada random bytelı veriler ürettik ve şifrelettik
        for i in range(num_ciphertexts):
            random_bytes = os.urandom(random.randint(100, 500))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin", mode="wb") as temp_file:
                temp_file.write(random_bytes)
                temp_file_path = temp_file.name
            #Şiferleme ve çözme işlemleri uyguluyoruz
            print(f"🔐 [{i+1}/{num_ciphertexts}] Dosya şifreleniyor: {os.path.basename(temp_file_path)}")
            aes_encrypt_file(temp_file_path, shared_secret_key)
            encrypted_path = temp_file_path.replace(".bin", ".enc")
            encrypted_path = os.path.join("encrypted_files", os.path.basename(encrypted_path))
            aes_decrypt_file(encrypted_path,shared_secret_key)
            #Temp fileları siliyoruz
            os.remove(temp_file_path)

            time.sleep(random.uniform(0.5, 3.0))
            
        #Loglardan özellik çıkarımı ve saldırı tespiti
        print("\n📊 Özellikler çıkarılıyor...")
        features = extract_cumulative_features()
        for k, v in features.items():
            print(f"{k}: {v}")

        print("\n🧠 Saldırı tespiti yapılıyor...")
        result = defineattack(features)
        print(f"\n🎯 Sonuç: {'🛑 Saldırı' if result =='Saldırı' else '✅ Normal'}")
        return result

    except Exception as e:
        print(f"❌ Simülasyon sırasında hata oluştu: {e}")
        return "Hata"


import tempfile
import time
import random



def simulate_cca_behavior_and_detect_attack(num_operations=5):
    print("🔍 CCA Simülasyonu başlatılıyor...")
    kem_alg = get_current_kem_alg()


    corrupt_dir = "corrupted_files"
    os.makedirs(corrupt_dir, exist_ok=True)

    try:
        with oqs.KeyEncapsulation(kem_alg) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            ciphertext, shared_secret_key = kem.encap_secret(public_key)

        for i in range(num_operations):
            # Rastgele plaintext oluştur
            plaintext = os.urandom(random.randint(100, 500))

            # Geçici dosyaya yaz
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin", mode="wb") as pt_file:
                pt_file.write(plaintext)
                pt_path = pt_file.name

            print(f"🔐 [{i+1}/{num_operations}] Dosya şifreleniyor: {os.path.basename(pt_path)}")
            aes_encrypt_file(pt_path, shared_secret_key)
            
            # Şifreli dosya yolu
            encrypted_path = pt_path.replace(".bin", ".enc")
            encrypted_path = os.path.join("encrypted_files", os.path.basename(encrypted_path))

            # Şifreli dosyayı boz
            with open(encrypted_path, "rb") as f:
                encrypted_data = f.read()

            corrupted_data = corrupt_bytes(encrypted_data, num_bytes=5)
            corrupted_path = os.path.join(corrupt_dir, os.path.basename(encrypted_path) + ".corrupt")

            with open(corrupted_path, "wb") as f:
                f.write(corrupted_data)

            print(f"🧨 [{i+1}/{num_operations}] Bozuk dosya oluşturuldu: {os.path.basename(corrupted_path)}")

            # Şimdi bozulan dosyayı çözmeye çalış
            print(f"🔑 [{i+1}/{num_operations}] Dosya şifresi çözülüyor: {os.path.basename(corrupted_path)}")
            try:
                if  aes_decrypt_file(corrupted_path, shared_secret_key):
                    print("❗ Bozuk dosya çözüldü! Bu istenmeyen bir durum.")
            except Exception as e:
                print(f"✅ Bozuk dosya çözülemedi, beklenen hata: {e}")

            # Temizleme
            os.remove(pt_path)
            os.remove(encrypted_path)
            os.remove(corrupted_path)

            # Rastgele bekle (1-4 sn)
            time.sleep(random.uniform(1.0, 4.0))

        print("\n📊 Özellikler çıkarılıyor...")
        features = extract_cumulative_features()
        for k, v in features.items():
            print(f"{k}: {v}")

        print("\n🧠 Saldırı tespiti yapılıyor...")
        result = defineattack(features)
        print(f"\n🎯 Sonuç: {'🛑 Saldırı' if result == 'Saldırı' else '✅ Normal'}")
        return result

    except Exception as e:
        print(f"❌ Simülasyon sırasında hata oluştu: {e}")
        return "Hata"

def simulate_brute_force_decryption_and_detect_attack(num_attempts=30):
    import tempfile
    import time
    import os
    import random
    from Crypto.Random import get_random_bytes
    import oqs

    print("🔓 Brute Force Decryption Simülasyonu başlatılıyor...")

    try:
        # 1. Doğru anahtarla bir dosya şifrele (kurban dosya)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as original_file:
            original_file.write("This is the real secret message.")
            original_file_path = original_file.name

        kem_alg = get_current_kem_alg()

        with oqs.KeyEncapsulation(kem_alg) as kem:
            public_key = kem.generate_keypair()
            ciphertext, correct_shared_key = kem.encap_secret(public_key)

        aes_encrypt_file(original_file_path, correct_shared_key)
        
        encrypted_path=os.path.join("encrypted_files", str(os.path.basename(original_file_path)).replace("txt","enc"))
        os.remove(original_file_path)  # Şifresiz dosyayı sil
        # 2. Brute force: Yanlış anahtarlarla çözmeye çalış
        for i in range(num_attempts):
            fake_key = get_random_bytes(len(correct_shared_key))

            try:
                aes_decrypt_file(encrypted_path, fake_key)
                print(f"❌ [{i+1}/{num_attempts}] Başarısız decrypt denemesi")
            except Exception:
                pass

            time.sleep(random.uniform(0.01, 0.05))  # Rastgele bekleme ile tempo

        # 3. Doğru anahtarla çözüm
        try:
            aes_decrypt_file(encrypted_path, correct_shared_key)
            print("✅ Gerçek anahtarla başarıyla çözüldü.")
        except Exception:
            print("❌ Doğru anahtarla çözümde hata!")

        # 4. Özellik çıkarımı ve saldırı tespiti
        print("\n📊 Özellikler çıkarılıyor...")
        features = extract_cumulative_features()
        for k, v in features.items():
            print(f"{k}: {v}")

        print("\n🧠 Saldırı tespiti yapılıyor...")
        result = defineattack(features)
        print(f"\n🚨 Brute Force Tespit Sonucu: {result}")

        return result

    except Exception as e:
        print(f"❌ Simülasyon hatası: {e}")
        return "Hata"

import os
import random
import time
import tempfile
from Crypto.Random import get_random_bytes

def simulate_kpa_behavior_and_detect_attack(num_operations=5):
    print("🧪 KPA Simülasyonu başlatılıyor...")

    try:
        # Sabit bir plaintext içeriği (saldırgan bunu biliyor)
        known_plaintext = b"This is a known plaintext content used in multiple encryptions."
        kem_alg = get_current_kem_alg()

        with oqs.KeyEncapsulation(kem_alg) as kem:
            public_key = kem.generate_keypair()
            _, shared_secret_key = kem.encap_secret(public_key)

        for i in range(num_operations):
            # Her seferinde aynı içerikle geçici dosya oluştur
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as pt_file:
                pt_file.write(known_plaintext)
                pt_path = pt_file.name

            print(f"🔐 [{i+1}/{num_operations}] Aynı içerik şifreleniyor: {os.path.basename(pt_path)}")
            aes_encrypt_file(pt_path, shared_secret_key)

            os.remove(pt_path)  # plaintext dosyayı hemen sil

            time.sleep(random.uniform(0.3, 1.2))  # aralıklı denemeler

        print("\n📊 Özellikler çıkarılıyor...")
        features = extract_cumulative_features()
        for k, v in features.items():
            print(f"{k}: {v}")

        print("\n🧠 Saldırı tespiti yapılıyor...")
        result = defineattack(features)
        print(f"\n🎯 KPA Tespit Sonucu: {'🛑 Saldırı' if result == 'Saldırı' else '✅ Normal'}")
        return result

    except Exception as e:
        print(f"❌ Simülasyon hatası: {e}")
        return "Hata"
#Random forest saldırı tespit modellerinde kullandığımız öznitelikler CPA,COA,CCA,Brute Force,KPA
listcpafeatures= [
    "encryption_count",
    "encrypts_per_minute",
    "encrypt_time_gap_min_sec",
    "encrypt_time_gap_avg_sec",
    "encrypt_time_gap_max_sec",
    "avg_encrypt_size_bytes",
    "avg_encrypt_entropy",
    "duplicate_hash_count",
    "unique_hash_count",
    "saved_encrypted_file_count"
]
listcoafeatures= ["duplicate_hash_count",
    "unique_hash_count",
    "most_common_hash_count",
    "avg_encrypt_entropy",
    "avg_encrypt_size_bytes",
    "saved_encrypted_file_count"]
listccafeatures=[ "decryption_count",
        "decrypts_per_minute",
        "avg_decrypt_duration_sec",
        "avg_decrypt_size_bytes",
        "avg_decrypt_entropy",
        "decrypt_success_ratio",
        "decrypt_time_gap_min_sec",
        "decrypt_time_gap_avg_sec",
        "decrypt_time_gap_max_sec",
        "duplicate_hash_count",
        "unique_hash_count",
        "most_common_hash_count",
        "saved_encrypted_file_count"]
listbruteforcefeatures=[
    "decryption_count",
    "decrypt_success_ratio",
    "decrypts_per_minute",
    "avg_decrypt_duration_sec",
    "decrypt_time_gap_min_sec",
    "decrypt_time_gap_avg_sec",
    "decrypt_time_gap_max_sec",
    "avg_decrypt_entropy"
]   
listkpafeatures=[ 
        "encryption_ratio",
        "duplicate_hash_count",
        "most_common_hash_count",
        "unique_hash_count",
        "avg_encrypt_entropy",
        "encrypt_time_gap_avg_sec",
        "avg_encrypt_duration_sec",
        "avg_encrypt_size_bytes"]
import pickle
import pandas as pd #burada hazır modelleri yüklüyoruz
from sklearn.ensemble  import RandomForestClassifier
def defineattack(data):
  attacknamelist=['cpa','coa','cca','bruteforce','kpa']
  attackfeaturelist=[listcpafeatures,listcoafeatures,listccafeatures,listbruteforcefeatures,listkpafeatures]

  list=[]
  #Burada saldırı isimleri ve eğitilmiş model dosyalrının listesi var

  for i in attacknamelist:
    dosya_yolu = f'models\{i}_model.pkl'

      # Modeli yükle

    with open(dosya_yolu, 'rb') as dosya:
        model = pickle.load(dosya)
      # Modeli yüklüyoruz
    temp_df = pd.DataFrame([data]) #df oluşturma
    #Loglardan alınan ve çıkartılan özellikler ilgili modele göre öznitelikleri seçilir

    df = temp_df[attackfeaturelist[attacknamelist.index(i)]]
    #İlgili özellik değerleri modellere verilir ve tahminleme

    tahmin = model.predict(df)
    print(i,tahmin)
    olasilik = model.predict_proba(df)[0][1]  # Sınıf 1 (saldırı) olasılığı
    print(olasilik)

    list.append(tahmin)

  if 1 in list:
    return "Saldırı"
  else :
    return "Normal"



def parse_log_line(line):
    #Yazılan her logu split edip ayrıştırı ve zaman damgası operasyon adı vb. gibi bir dictionarye dönüştürür
    try:
        parts = line.strip().split('|')
        if len(parts) < 3:
            return None
        timestamp_str = parts[0].strip()
        operation_type = parts[1].strip()
        filename = parts[2].strip()
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return {
            "timestamp": timestamp,
            "operation_type": operation_type,
            "filename": filename,
            "raw_line": line.strip()
        }
    except Exception as e:
        return None


from collections import Counter

def extract_cumulative_features(log_path="logs/operation_log.txt"):
    #Logları kaydettiğimiz bir dosya satırları okur
    if not os.path.exists(log_path):
        print("Log dosyası bulunamadı:", log_path)
        return {}

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    hashes = []
    sizes = []
    #Burada ise logların kayıtları parse edilip incelenir ve kümülatif bir hesap yapılır
    for line in lines:
        parsed = parse_log_line(line)
        if parsed is None:
            continue

        data = line.strip().split('|')
        info = {}
        for d in data[3:]:
            d = d.strip()
            if d.startswith("Boyut:"):
                try:
                    size = int(d.split()[1])
                    info["size"] = size
                    sizes.append(size)
                except:
                    info["size"] = None
            elif d.startswith("Entropi:"):
                try:
                    info["entropy"] = float(d.split()[1])
                except:
                    info["entropy"] = None
            elif d.startswith("Süre:"):
                try:
                    info["duration"] = float(d.split()[1])
                except:
                    info["duration"] = None
            elif d.startswith("Çözüm Başarısı:"):
                val = d.split(":", 1)[1].strip()  # ':' dan sonraki kısmı al
                info["decrypt_success"] = (val == "Evet")
            elif d.startswith("SHA256:"):
                hash_val = d.split()[1]
                info["hash_value"] = hash_val
                hashes.append(hash_val)

        parsed.update(info)
        entries.append(parsed)

    total_ops = len(entries)
    encrypt_entries = [e for e in entries if e['operation_type'] == "AES Şifreleme"]
    decrypt_entries = [e for e in entries if e['operation_type'] == "AES Çözme"]

    encrypt_count = len(encrypt_entries)
    decrypt_count = len(decrypt_entries)

    # Hash analizleri
    hash_counts = Counter(hashes)
    unique_hash_count = len(hash_counts)
    most_common_hash, most_common_hash_count = hash_counts.most_common(1)[0] if hash_counts else (None, 0)
    duplicate_hash_count = sum(1 for h, c in hash_counts.items() if c > 1)

    encrypt_ratio = encrypt_count / total_ops if total_ops > 0 else 0
    decrypt_ratio = decrypt_count / total_ops if total_ops > 0 else 0

    total_encrypt_duration = sum(e.get("duration", 0) or 0 for e in encrypt_entries)
    avg_encrypt_duration = total_encrypt_duration / encrypt_count if encrypt_count > 0 else 0

    total_decrypt_duration = sum(e.get("duration", 0) or 0 for e in decrypt_entries)
    avg_decrypt_duration = total_decrypt_duration / decrypt_count if decrypt_count > 0 else 0

    avg_encrypt_size = sum(e.get("size", 0) or 0 for e in encrypt_entries) / encrypt_count if encrypt_count > 0 else 0
    avg_decrypt_size = sum(e.get("size", 0) or 0 for e in decrypt_entries) / decrypt_count if decrypt_count > 0 else 0

    avg_encrypt_entropy = sum(e.get("entropy", 0) or 0 for e in encrypt_entries) / encrypt_count if encrypt_count > 0 else 0
    avg_decrypt_entropy = sum(e.get("entropy", 0) or 0 for e in decrypt_entries) / decrypt_count if decrypt_count > 0 else 0

    successful_decrypts = sum(1 for e in decrypt_entries if e.get("decrypt_success") == True)
    decrypt_success_ratio = successful_decrypts / decrypt_count if decrypt_count > 0 else None

    if total_ops > 1:
        times = [e['timestamp'].timestamp() for e in entries]
        duration_min = (max(times) - min(times)) / 60
        if duration_min == 0:
            duration_min = 1
    else:
        duration_min = 1

    encrypts_per_minute = encrypt_count / duration_min
    decrypts_per_minute = decrypt_count / duration_min
    

    features = {
        "total_operations": total_ops,
        "encryption_count": encrypt_count,
        "decryption_count": decrypt_count,
        "encryption_ratio": encrypt_ratio,
        "decryption_ratio": decrypt_ratio,
        "encrypts_per_minute": encrypts_per_minute,
        "decrypts_per_minute": decrypts_per_minute,
        "avg_encrypt_duration_sec": avg_encrypt_duration,
        "avg_decrypt_duration_sec": avg_decrypt_duration,
        "avg_encrypt_size_bytes": avg_encrypt_size,
        "avg_decrypt_size_bytes": avg_decrypt_size,
        "avg_encrypt_entropy": avg_encrypt_entropy,
        "avg_decrypt_entropy": avg_decrypt_entropy,
        "decrypt_success_ratio": decrypt_success_ratio,
        "unique_hash_count": unique_hash_count,
        "most_common_hash_count": most_common_hash_count,
        "duplicate_hash_count": duplicate_hash_count,
    }

        # === Kaydedilmiş .enc uzantılı dosya sayısı ===
    saved_encrypted_files = [e for e in entries if e['operation_type'] == "Dosya Kaydetme" and e['filename'].endswith(".enc")]
    features["saved_encrypted_file_count"] = len(saved_encrypted_files)

        # === İki AES şifreleme işlemi arasındaki zaman farkı ===
    encrypt_times = [e["timestamp"] for e in encrypt_entries]
    encrypt_deltas = [
        (encrypt_times[i] - encrypt_times[i - 1]).total_seconds()
        for i in range(1, len(encrypt_times))
    ]
    if encrypt_deltas:
        features["encrypt_time_gap_avg_sec"] = sum(encrypt_deltas) / len(encrypt_deltas)
        features["encrypt_time_gap_max_sec"] = max(encrypt_deltas)
        features["encrypt_time_gap_min_sec"] = min(encrypt_deltas)
    else:
        features["encrypt_time_gap_avg_sec"] = None
        features["encrypt_time_gap_max_sec"] = None
        features["encrypt_time_gap_min_sec"] = None

    # === İki AES çözme işlemi arasındaki zaman farkı ===
    decrypt_times = [e["timestamp"] for e in decrypt_entries]
    decrypt_deltas = [
        (decrypt_times[i] - decrypt_times[i - 1]).total_seconds()
        for i in range(1, len(decrypt_times))
    ]
    if decrypt_deltas:
        features["decrypt_time_gap_avg_sec"] = sum(decrypt_deltas) / len(decrypt_deltas)
        features["decrypt_time_gap_max_sec"] = max(decrypt_deltas)
        features["decrypt_time_gap_min_sec"] = min(decrypt_deltas)
    else:
        features["decrypt_time_gap_avg_sec"] = None
        features["decrypt_time_gap_max_sec"] = None
        features["decrypt_time_gap_min_sec"] = None

    return features



def compute_hash(data_bytes):
    """Verilen byte dizisinin SHA-256 hash'ini döner."""
    return hashlib.sha256(data_bytes).hexdigest()

def calculate_entropy(data_bytes):
    """Verilen byte dizisinin Shannon entropisini (bit cinsinden) hesaplar."""
    if not data_bytes:
        return 0
    byte_counts = np.bincount(np.frombuffer(data_bytes, dtype=np.uint8), minlength=256)
    probs = byte_counts / np.sum(byte_counts)
    return shannon_entropy(probs, base=2)


def write_log(operation_type, filename, size=0, description="", 
              entropy=None, hash_value=None, decrypt_success=None, encrypt_success=None,
              processing_time=None, extra_info=None):
    log_dir = "logs"
    ensure_directory_exists(log_dir) #Burada ise yapılan her işlemi loglara kaydedilir

    log_path = os.path.join(log_dir, "operation_log.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #her işlemden sonra writelog fonksiyonu kullanılır ve ilgili parametereler kaydedilir
    line = f"{now} | {operation_type} | {filename} | Boyut: {size} byte"

    if hash_value is not None:
        line += f" | SHA256: {hash_value}"
    if entropy is not None:
        line += f" | Entropi: {entropy:.4f}"
    if decrypt_success is not None:
        line += f" | Çözüm Başarısı: {'Evet' if decrypt_success else 'Hayır'}"
    if processing_time is not None:
        line += f" | Süre: {processing_time:.4f} sn"
    if encrypt_success is not None:
        line += f" | Şifreleme Başarısı: {'Evet' if encrypt_success else 'Hayır'}"
    if extra_info:
        line += f" | {extra_info}"
    if description:
        line += f" | {description}"

    line += "\n"

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(line)

import numpy as np
#Bu kısım kyber anahtar kapsüllemede ciphertexti maskelemek için kullanılan serpiştirme algoritması
def interleave_merge(v1, v2):
    merged = []
    max_len = max(len(v1), len(v2))
    for i in range(max_len):
        if i < len(v1):
            merged.append(v1[i])
        if i < len(v2):
            merged.append(v2[i])
    return merged
#Bu kodda maskelenmiş ciphertexti geri döndüren  kod
def interleave_split(merged, len_v1, len_v2):
    v1, v2 = [], []
    i = j = 0
    for k in range(len(merged)):
        if k % 2 == 0 and i < len_v1:
            v1.append(merged[k])
            i += 1
        elif j < len_v2:
            v2.append(merged[k])
            j += 1
        elif i < len_v1:
            v1.append(merged[k])
            i += 1
    return v1, v2

# kemalg,public,secretkey ve ciphertext dosyalarına ilave yapar
def append_hex_line(path, data_hex):
    with open(path, "a") as f:
        f.write(data_hex + "\n")
# ilgili dosyalrı satır satır  okuma
def read_all_lines_hex(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]



def detect_file_type(file_path):
    #mimetypes ile dosya türünü tespit eder
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        return b'UNKN'
    if mime_type.startswith('text'):
        return b'TEXT'
    elif mime_type.startswith('image'):
        return b'IMAG'
    elif mime_type.startswith('audio'):
        return b'AUDI'
    else:
        return b'UNKN'
#ilgili dizinin var olup olmadığını kontrol eder
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
#dizindeki dosyaları listeler
def list_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# === AES Şifreleme ===

def aes_encrypt_file(input_path, key, output_dir="encrypted_files"):
    try:
        import time
        start = time.time() #duration hesabı için
        with open(input_path, 'rb') as f:
            data = f.read()

        file_type_header = detect_file_type(input_path)
        data_with_header = file_type_header + data #geri dönüşümde zorluk çıkmasın diye dosya tipini aldık

        nonce = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data_with_header)

        duration = time.time() - start 
        ensure_directory_exists(output_dir)

        original_name = os.path.splitext(os.path.basename(input_path))[0]
        encrypted_path = os.path.join(output_dir, original_name + ".enc") #şifreleme ve şifrelenmiş dosyayı kaydetme

        with open(encrypted_path, 'wb') as f:
            f.write(nonce + ciphertext+tag)
        

        print(f"🔒 Şifrelendi: {encrypted_path}")
        entropy_val = calculate_entropy(ciphertext)#entropi hesabı
        hash_val = compute_hash(ciphertext)

        write_log("AES Şifreleme", encrypted_path, len(ciphertext),
                "Dosya AES ile şifrelendi",
                entropy=entropy_val,
                hash_value=hash_val,
                processing_time=duration)
        write_log("Dosya Kaydetme", encrypted_path, os.path.getsize(encrypted_path),
                "Şifrelenmiş dosya başarıyla kaydedildi")
        return True
    except Exception as e:
        duration = time.time() - start
        #şifreleme başarısız olma durumunda log kaydı
        write_log("AES Şifreleme", input_path, 0,
                  f"Şifreleme başarısız: {e}",
                  decrypt_success=False,
                  processing_time=duration)
        print(f"❌ Şifreleme başarısız: {e}")
        return False



# === AES Çözme ===

def aes_decrypt_file(encrypted_path, key, output_dir="decrypted_files"):
    try:
        import time
        start = time.time()
        with open(encrypted_path, 'rb') as f:
            file_content = f.read()
        nonce = file_content[:12]            # İlk 12 byte nonce
        tag = file_content[-16:]             # Son 16 byte tag
        ciphertext = file_content[12:-16]   # Aradaki kısım ciphertext
        
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        #Çözme işlemi yapılıyor
        decrypted_data =  cipher.decrypt_and_verify(ciphertext, tag)
        duration = time.time() - start
        file_type_header = decrypted_data[:4] #dosya türü buradan geri alınıyor
        original_data = decrypted_data[4:]
        entropy_val = calculate_entropy(original_data) #entropi hesabı
        hash_val = compute_hash(original_data) #hashleme
        extension_map = {
            b'TEXT': '.txt',
            b'IMAG': '.jpg',
            b'AUDI': '.mp3',
            b'UNKN': '.bin'
        }

        extension = extension_map.get(file_type_header, '.bin') #map ile dosya türünü dönüşümü
        original_filename = os.path.basename(encrypted_path).replace(".enc", "")
        name_part, _ = os.path.splitext(original_filename)
        output_filename = f"{name_part}_decrypted{extension}"

        ensure_directory_exists(output_dir)
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'wb') as f:
            f.write(original_data)
        #Çözüm başarılı kaydı
        print(f"🔓 Çözüldü: {output_path}")
        write_log("Dosya Kaydetme", output_path, os.path.getsize(output_path),
                "Çözülmüş dosya başarıyla kaydedildi")
        
        # AES çözme işlemi logu (entropi, hash, süre dahil)
        write_log("AES Çözme", output_path, len(original_data),
                "Dosya başarıyla çözüldü",
                entropy=entropy_val,
                hash_value=hash_val,
                decrypt_success=True,
                processing_time=duration)
        return True
    except Exception as e:
        duration = time.time() - start
        #Çözüm başarısız kaydı
        write_log("AES Çözme", encrypted_path, 0,
                  f"Çözme başarısız: {e}",
                  decrypt_success=False,
                  processing_time=duration)
        print(f"❌ Çözme başarısız: {e}")
        return False




# === Ana Program ===

if __name__ == "__main__":
    print("1) Yeni anahtar ve ciphertext oluştur, dosya AES ile şifrele")
    print("2) Dışarıdan ciphertext gir, CSV'de ara, secret key ile çöz ve AES dosya çöz")
    kem_alg = get_current_kem_alg()
    choice = input("Seçiminiz (1 veya 2): ").strip()

    if choice == "1":
        # Yeni anahtar oluştur, ciphertext oluştur ve csv'ye ekle
        with oqs.KeyEncapsulation(kem_alg) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            #Anahtar üretimi ve csv dosyalarına kaydetme
            append_hex_line("public_key.csv", public_key.hex())
            append_hex_line("secret_key.csv", secret_key.hex())
            # Anahtar kapsülleme
            ciphertext, shared_secret_enc = kem.encap_secret(public_key)
            #Kapsüllenmiş anahtarın ciphertextini maskeleme
            v1=list(str(ciphertext.hex()))
            v2 = list("abcd1234ef567890abcd1234ef567890")
            #Kullanıcıya maskeli ciphertext verme
            merged = interleave_merge(v1, v2)
            print("Karışık:", "".join(merged))
            
            #İlgili ciphertext ve kem_alg kaydetme
            append_hex_line("ciphertext.csv", ciphertext.hex())
            append_hex_line("kem_alg.csv", kem_alg)
            
        
        # Şifrelenecek dosya seçimi
        path = input("AES ile şifrelenecek dosyanın yolunu girin: ").strip()
        if os.path.isfile(path):
            size = os.path.getsize(path)
            #Dosya yükleme logu alınır
            write_log("Dosya Yükleme", os.path.basename(path), size, "Kullanıcı tarafından şifrelenecek dosya seçildi")
            features = extract_cumulative_features()
            #logdan özellik çıkarımı
            print("\n--- Kümülatif Özellikler ---")
            for k, v in features.items():
                print(f"{k}: {v}")

            aes_encrypt_file(path, shared_secret_enc)
            #Şifreleme 
            features = extract_cumulative_features()
            print("\n--- Kümülatif Özellikler ---")
            for k, v in features.items():
                print(f"{k}: {v}")
            checkpoint=defineattack(features)
            print(checkpoint)
            #Özellik çıkarımı ve saldırı kontrolü
        
        else:
            print("❌ Geçersiz dosya yolu!")

        if checkpoint=='Saldırı':
            kem_alg=upgrade_kem_alg(kem_alg)
            clear_log_file()
            #Saldırı varsa anahtar seviyesi yükselsin

    elif choice == "2":
        # Dışarıdan ciphertext hex al, csv'de ara, karşılık secret key ile decapsulate yap
        hex_input = input("Aranacak ciphertext hex değerini girin:\n").strip()
        
        
        ciphertext_list = read_all_lines_hex("ciphertext.csv")#Tüm satırlar okunur
        secretkey_list = read_all_lines_hex("secret_key.csv")
        ayristirilan_v1, ayristirilan_v2 = interleave_split(hex_input, len(hex_input)-32, 32)
        #Maskeli ciphertext eski haline dönüştürülür bir string ile karıştırılmıştı
        hex_input="".join(ayristirilan_v1)
        if hex_input not in ciphertext_list:
            print("❌ Ciphertext dosyada bulunamadı.")#Ciphertext bulundu mu sorgusu
            exit()

        index = ciphertext_list.index(hex_input)#Bulduysak ilgili dosyada indexi buluruz
        kemalg_list = read_all_lines_hex("kem_alg.csv")#Diğer dosyalarda aynı indexteki yerlere bakarak anahtarı decapsüle deriz
        secret_key_hex = secretkey_list[index]
        secret_key = bytes.fromhex(secret_key_hex)
        kem_alg = kemalg_list[index]
        ciphertext_bytes = bytes.fromhex(hex_input)#Byte dönüşümü

        with oqs.KeyEncapsulation(kem_alg, secret_key=secret_key) as kem:
            shared_secret_dec=kem.decap_secret(ciphertext_bytes)#Decapsüle işlemi
        print(f"✅ Ciphertext bulundu ve secret key ile çözüldü.")
        print(f"Shared secret: {shared_secret_dec.hex()}")#Shared secret bulundu

        # AES çözme için dosya yolu al
        path = input("AES ile çözülecek dosya yolunu girin (veya boş bırak çık): ").strip()
        
        if path:
            
            if os.path.isfile(path):
                aes_decrypt_file(path, shared_secret_dec)
                features = extract_cumulative_features()
                print("\n--- Kümülatif Özellikler ---")
                for k, v in features.items():
                    print(f"{k}: {v}")
                checkpoint=defineattack(features)# Dosya yolu alınırken  iken saldırı kontrolü
                print(checkpoint)
            
            else:
                print("❌ Geçersiz dosya yolu!")
        if checkpoint=='Saldırı':
            kem_alg=upgrade_kem_alg(kem_alg)
            clear_log_file()
            #Saldırı varsa anahtar seviyesi yükselsin
    elif choice == "3": 
        if "Saldırı"==simulate_cpa_behavior_and_detect_attack():
            kem_alg = upgrade_kem_alg(kem_alg)#Burada saldırı simülasyonları mevcut
            clear_log_file()
    elif choice == "4":
        if "Saldırı"==simulate_coa_behavior_and_detect_attack():
            kem_alg = upgrade_kem_alg(kem_alg)
            clear_log_file()
    elif choice=="5":
        if "Saldırı"==simulate_cca_behavior_and_detect_attack():
            kem_alg = upgrade_kem_alg(kem_alg)
            clear_log_file()
    elif choice=="6":
        if "Saldırı"==simulate_brute_force_decryption_and_detect_attack():
            kem_alg = upgrade_kem_alg(kem_alg)
            clear_log_file()
    elif choice=="7":
        if "Saldırı"==simulate_kpa_behavior_and_detect_attack():
            kem_alg = upgrade_kem_alg(kem_alg)
            clear_log_file()
    elif choice=="Dosya Sıkıştırma":
        #Dosya sıkıştırma kısmı
        path = input("Sıkıştırma yapılacak dosya yolunu girin ").strip()
        #Dosya yolu istenir ve dosyanın varlığı kontrol edilir
        if os.path.isfile(path):
            size = os.path.getsize(path)
            write_log("Dosya Sıkıştırma İçin Yükleme", os.path.basename(path), size, "Kullanıcı tarafından sıkıştırılcak dosya seçildi")
            #Loglama işlemi yapılır 
            #detect file ile tür tespiti yapılır ve ilgili sıkıştırma fonksiyonuna gönderilir
            filetype=detect_file_type(path)
            print(filetype)
            if filetype==b"IMAG":
                compress_adaptively_image(path)#resim sıkıştırma
                
            elif filetype==b"TEXT":
                output_zip_file = f"{path}_compressed.zip".replace(".txt","")

                encode_text_and_create_zip(path, output_zip_file)
                #metin sıkıştırma  eğer çözmek istersek decode kodu da bulunuyor
                """original = decode_from_zip(output_zip_file)
                print("\nZIP'ten açılıp decode edilmiş metin:\n")
                print(original) """
            elif filetype==b"AUDI":
                output_file = f"{path}.mp3".replace(".wav","")
                # ses için sıkıştırma 
                duration = get_audio_duration(path)# duration hesabı
                embedding = extract_vggish_embedding(path)#embedding hesabı
                bitrate = compute_bitrate_from_embedding(embedding, duration) # bitrat hesabı

                print(f"Seçilen bitrate: {bitrate} kbps (Ses süresi: {duration:.2f} saniye)")
                compress_audio_ffmpeg(path, output_file, bitrate, duration)  #ses sıkıştırma
            else:
                print("Dosya tipi tespit edilemedi.")
    else:
        print("❌ Geçersiz seçim.")
