# 🫁 Vision-GPT2: Chest X-Ray Report Generation

Göğüs röntgeni görüntülerinden otomatik radyoloji raporu üretimi için Vision-GPT2 tabanlı derin öğrenme modeli.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Model Mimarisi](#-model-mimarisi)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Eğitim](#-eğitim)
- [Sonuçlar](#-sonuçlar)
- [Dosya Yapısı](#-dosya-yapısı)

## 🎯 Genel Bakış

Bu proje, göğüs röntgenlerinden otomatik olarak **Findings** ve **Impression** bölümlerini içeren yapılandırılmış radyoloji raporları üretebilen bir Vision-Language model içermektedir.

### Özellikler

- ✅ CNN tabanlı görüntü kodlayıcı (DenseNet121)
- ✅ GPT-2 tabanlı dil çözücü
- ✅ Cross-attention mekanizması
- ✅ Streamlit web arayüzü
- ✅ Gerçek zamanlı rapor üretimi

## 🏗 Model Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    Göğüs Röntgeni Görüntüsü                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Görüntü Ön İşleme (384x384, Normalize)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                CNN Encoder (DenseNet121)                     │
│                  Görsel Özellik Çıkarımı                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Cross-Attention Module                      │
│              (Görsel ↔ Metinsel Hizalama)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPT-2 Dil Çözücü                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Üretilen Radyoloji Raporu                       │
│              (Findings + Impression)                         │
└─────────────────────────────────────────────────────────────┘
```

## 💻 Kurulum

### Gereksinimler

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU için)

### Adımlar

```bash
# 1. Repo'yu klonla
git clone https://github.com/CengizhanKARAGOZ/vision-gpt2-cxr-report.git
cd vision-gpt2-cxr-report

# 2. Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
.\venv\Scripts\activate  # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Model ağırlıklarını indir (eğitilmiş model)
# weights/ klasörüne best_vgpt2.pt dosyasını koy
```

## 🚀 Kullanım

### Streamlit Web Arayüzü

```bash
cd app
streamlit run streamlit_app.py
```

Tarayıcıda `http://localhost:8501` adresine git.

### Python API

```python
from app.model_infer import load_model, build_transform, preprocess_pil
from PIL import Image

# Model yükle
model, tokenizer = load_model("weights/best_vgpt2.pt", device="cpu")
transform = build_transform(img_size=384)

# Görüntü yükle ve işle
image = Image.open("xray.jpg")
x = preprocess_pil(image, transform)

# Rapor üret
report = model.generate(
    x, tokenizer,
    prompt="Findings: The",
    max_new_tokens=100,
    temperature=0.70,
    repetition_penalty=1.20
)

print(report)
```

## 🎓 Eğitim

### Veri Seti

Model, [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) veri seti üzerinde eğitilmiştir.

- **Eğitim örnekleri:** ~42,000
- **Doğrulama örnekleri:** ~313

### Kaggle'da Eğitim

```bash
# Kaggle notebook'ta çalıştır
python train/train.py
```

### Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 8e-5 |
| Weight Decay | 0.01 |
| Batch Size | 8 |
| Gradient Accumulation | 4 |
| Epochs | 4 |
| Image Size | 384x384 |
| Max Sequence Length | 256 |
| Label Smoothing | 0.1 |
| LR Schedule | Cosine Annealing |
| Mixed Precision | FP16 |

### Eğitim Sonuçları

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 2.0574 | 1.8070 |
| 2 | 1.7499 | 1.7235 |
| 3 | 1.6880 | 1.6975 |
| 4 | 1.6626 | 1.6854 |

## 📊 Sonuçlar

### Örnek Çıktılar

**Normal Bulgular:**
```
Findings: The lungs are clear without focal consolidation, pleural 
effusion, or pneumothorax. Heart and mediastinal silhouettes are normal.

Impression: No acute cardiopulmonary process.
```

**Patolojik Bulgular:**
```
Findings: There is increased opacity in the right lower lobe consistent 
with consolidation. Small right pleural effusion is noted.

Impression: Right lower lobe pneumonia with small pleural effusion.
```

### Model Performansı

- ✅ Normal vakalarda yüksek tutarlılık
- ✅ Standart radyoloji terminolojisi kullanımı
- ✅ Findings-Impression yapısal uyumu
- ⚠️ Patolojik detaylarda sınırlı spesifiklik

## 📁 Dosya Yapısı

```
vision-gpt2-cxr-report/
├── README.md                 
├── requirements.txt          # Python bağımlılıkları
├── train/
│   └── train.py             # Eğitim scripti
├── app/
│   ├── model_infer.py       # Model inference modülü
│   └── streamlit_app.py     # Web arayüzü
├── weights/
│   └── best_vgpt2.pt        # Eğitilmiş model (gitignore)
└── examples/
    └── sample_xray.jpg      # Örnek görüntü
```

## 📝 Notlar

- Bu model **demo amaçlıdır**, klinik kullanım için uygun değildir.
- Üretilen raporlar mutlaka uzman radyolog tarafından doğrulanmalıdır.
- Model, MIMIC-CXR veri setinin dağılımına göre eğitilmiştir.

## 📄 Lisans

MIT License

## 🙏 Teşekkürler

- [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.0.0/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [timm](https://github.com/huggingface/pytorch-image-models)

## 📚 Referanslar

1. Irvin, J., et al. (2019). CheXpert: A large chest radiograph dataset.
2. Radford, A., et al. (2019). Language models are unsupervised multitask learners.
3. Chen, Z., et al. (2022). Generating radiology reports via memory-driven transformer.