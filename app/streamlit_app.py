import os
import streamlit as st
import torch
from PIL import Image

from model_infer import load_model, build_transform, preprocess_pil

st.set_page_config(
    page_title="CXR Report Generator",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container - compact but readable */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1600px !important;
    }

    /* Title */
    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
        margin-top: 0 !important;
    }

    /* Image - controlled size */
    [data-testid="stImage"] img {
        max-height: 280px !important;
        width: auto !important;
        object-fit: contain !important;
    }

    /* Report boxes - readable */
    .report-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 12px;
    }

    .report-box.impression {
        border-left-color: #2196F3;
    }

    .report-title {
        font-size: 12px;
        font-weight: 700;
        color: #aaa;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .report-content {
        font-size: 15px;
        line-height: 1.5;
        color: #e8e8e8;
    }

    /* Button */
    .stButton > button {
        padding: 0.6rem 1rem !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }

    /* File uploader compact */
    [data-testid="stFileUploader"] section {
        padding: 0.6rem !important;
    }

    /* Hide elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Placeholder */
    .placeholder-box {
        background: #1a1a2e;
        border: 2px dashed #444;
        border-radius: 10px;
        padding: 30px 20px;
        text-align: center;
        color: #777;
        font-size: 14px;
    }

    .placeholder-box span {
        font-size: 32px;
        display: block;
        margin-bottom: 8px;
    }

    /* Sidebar compact */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }

    [data-testid="stSidebar"] .stSlider {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0 !important;
    }

    [data-testid="stSidebar"] .stTextInput {
        margin-bottom: 0 !important;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }

    [data-testid="stSidebar"] h4 {
        margin-bottom: 0.3rem !important;
        margin-top: 0 !important;
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🫁 Göğüs Röntgeni Rapor Üretici")

with st.sidebar:
    st.markdown("#### ⚙️ Ayarlar")
    ckpt_path = st.text_input("Checkpoint", value=os.path.join("weights", "best_vgpt2.pt"))
    img_size = st.selectbox("Boyut", [384, 512], index=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"{'🚀' if device == 'cuda' else '💻'} Cihaz: {device.upper()}")

    st.markdown("---")
    st.markdown("#### 🎛️ Parametreler")

    max_new_tokens = st.slider("Tokens", 60, 200, 100, 10,
                               help="Üretilecek maksimum token (kelime parçası) sayısı. 60-80: Kısa ve öz rapor, 100-120: Standart uzunluk, 150+: Detaylı rapor. Çok yüksek değerler tekrara yol açabilir.")
    temperature = st.slider("Temperature", 0.2, 1.5, 0.70, 0.05,
                            help="Modelin yaratıcılık/rastgelelik seviyesi. 0.2-0.5: Tutarlı ve güvenli çıktılar, 0.6-0.8: Dengeli (önerilen), 0.9+: Daha yaratıcı ama hallucination riski artar.")
    top_p = st.slider("Top-p", 0.5, 1.0, 0.90, 0.01,
                      help="Nucleus sampling parametresi. Model, toplam olasılığı bu değere ulaşan en olası kelimeler arasından seçim yapar. 0.9: Standart, 0.7: Daha odaklı, 1.0: Tüm kelimeler dahil.")
    top_k = st.slider("Top-k", 0, 200, 50, 1,
                      help="Her adımda en yüksek olasılıklı K kelime arasından seçim yapar. 0: Devre dışı (sadece top-p kullanılır), 50: Standart, 20-30: Daha tutarlı çıktı.")
    rep_penalty = st.slider("Rep Penalty", 1.0, 1.5, 1.25, 0.01,
                            help="Tekrar eden kelimelere uygulanan ceza. 1.0: Ceza yok, 1.1-1.2: Hafif ceza (önerilen), 1.3+: Güçlü ceza. Yüksek değerler 'Impression: ... Impression:' gibi tekrarları önler.")


@st.cache_resource
def cached_load(ckpt, dev):
    return load_model(ckpt, device=dev)


if not os.path.exists(ckpt_path):
    st.error(f"❌ Checkpoint bulunamadı: `{ckpt_path}`")
    st.stop()

model, tok = cached_load(ckpt_path, device)
tfm = build_transform(img_size)

if 'generated_text' not in st.session_state:
    st.session_state.generated_text = None
if 'findings' not in st.session_state:
    st.session_state.findings = ""
if 'impression' not in st.session_state:
    st.session_state.impression = ""

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Görüntü Yükle")
    uploaded_file = st.file_uploader("Yükle", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, use_container_width=True)

        if st.button("🔬 Rapor Üret", type="primary", use_container_width=True):
            x = preprocess_pil(pil_image, tfm).to(device)

            with st.spinner("Rapor üretiliyor..."):
                text = model.generate(
                    x, tok,
                    prompt="Findings: The",
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else None,
                    temperature=temperature,
                    repetition_penalty=rep_penalty
                )
                findings, impression = model.parse_findings_impression(text)
                st.session_state.generated_text = text
                st.session_state.findings = findings
                st.session_state.impression = impression
            st.rerun()
    else:
        st.markdown('<div class="placeholder-box"><span>🫁</span>Röntgen görüntüsü yükleyin</div>',
                    unsafe_allow_html=True)

with col2:
    st.subheader("📋 Üretilen Rapor")

    if st.session_state.generated_text:
        st.markdown(f"""
        <div class="report-box">
            <div class="report-title">📝 FINDINGS</div>
            <div class="report-content">{st.session_state.findings or '<em>Findings bulunamadı</em>'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="report-box impression">
            <div class="report-title">💡 IMPRESSION</div>
            <div class="report-content">{st.session_state.impression or '<em>Impression bulunamadı</em>'}</div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            st.download_button(
                "📥 TXT İndir",
                data=f"FINDINGS:\n{st.session_state.findings}\n\nIMPRESSION:\n{st.session_state.impression}".encode(),
                file_name="rapor.txt",
                use_container_width=True
            )
        with col_b:
            import json

            st.download_button(
                "📋 JSON İndir",
                data=json.dumps({
                    "findings": st.session_state.findings,
                    "impression": st.session_state.impression
                }, indent=2, ensure_ascii=False).encode(),
                file_name="rapor.json",
                use_container_width=True
            )
        with col_c:
            if st.button("🗑️ Temizle", use_container_width=True):
                st.session_state.generated_text = None
                st.session_state.findings = ""
                st.session_state.impression = ""
                st.rerun()

        with st.expander("Ham Çıktı"):
            st.code(st.session_state.generated_text, language=None)
    else:
        st.markdown(
            '<div class="placeholder-box"><span>📋</span>Görüntü yükleyin ve "Rapor Üret" butonuna tıklayın</div>',
            unsafe_allow_html=True)