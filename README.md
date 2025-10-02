**Yapay Zeka Tabanlı Veri Şifreleme ve Sıkıştırma Algoritması**

Bu proje; metin, ses ve görsel veriler ile çalışan yapay zeka tabanlı bir sistemin prototipidir (Proof of Concept).
Sistem özel yapay zeka modelleri kullanarak her veri türünü analiz eder ve bir kalite skoru üretir. Ardından en uygun sıkıştırma seviyesini önerir ve veriyi sıkıştırır.

Güvenlik tarafında ise:

Kyber (kuantuma dayanıklı şifreleme yöntemi)

AES (Advanced Encryption Standard)

kullanılarak veri korunur.

Ayrıca sistem log dosyalarını kontrol ederek  öznitelik çıkarımı yapar ve random foreste vererek olası saldırıları tespit eder ve şifreleme ayarlarını otomatik olarak günceller.

**🚀 Kurulum**
Gerekli Python Kütüphaneleri
pip install numpy
pip install pickle
pip install pandas
pip install scikit-learn
pip install nltk
pip install pycryptodome
pip install torch
pip install soundfile
pip install torchvggish
pip install scipy
pip install Pillow
pip install ftfy regex tqdm   # CLIP için önkoşullar
pip install git+https://github.com/openai/CLIP.git

**Diğer Bağımlılıklar**
🎵 MP3 Sıkıştırma için

Sıkıştırma işlemleri için ffmpeg kurulu olmalıdır.
Windows için önerilen kurulum yolu Chocolatey kullanmaktır.

Chocolatey Kurulumu (PowerShell’i yönetici olarak çalıştırın):

Set-ExecutionPolicy Bypass -Scope Process -Force; `
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
`
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))


ffmpeg Kurulumu:

choco install ffmpeg -y

**🔐 Liboqs (Kyber Şifreleme için)**

Projede liboqs kütüphanesinin Python sarmalayıcısı kullanılmaktadır.
Kurulum için ek bağımlılıklar gerekir:

CMake

Git

C Compiler (gcc, clang, MSVC vb.)

Python 3.x.x

Daha fazla detay: liboqs-python

Visual Studio Build Tools Kurulumu

Visual Studio Build Tools
 indirin.

Kurulumda C++ build tools seçin.

Windows 10 SDK ve CMake işaretli olsun.

Eğer C++ Build Tools kuruluysa Desktop development with C++ seçeneğini aktif edin.

Liboqs Kurulumu
cmake -S liboqs -B liboqs/build -DCMAKE_INSTALL_PREFIX="C:\liboqs" -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=ON


Eğer hata alırsanız alternatif komutu deneyin:

cmake -S liboqs -B liboqs/build -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX="C:/liboqs" -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=ON


Not: Parallel işlem sayısı bilgisayara göre ayarlanabilir (varsayılan 8).

**📂 Model Hakkında**

Sıkıştırma algoritması resim için png metin için txt ses içinde waw dosyalarında çalışmaktadır.
Saldırı tespit simülasyonu için modelleri çalıştırdığınız ortama yükleyin.

Şifreleme algoritmasının ürettiği ciphertext kopyalanarak saklanmalıdır.

Saldırı tespit için kullanılan modeller eğitilmiş ağırlıklar ve datasetlerle birlikte yüklenmelidir.

Random Forest eğitim kodları ayrı bir dosyada bulunmaktadır.

Doğrudan ağırlıkları kullanarak sistemi test edebilirsiniz.
