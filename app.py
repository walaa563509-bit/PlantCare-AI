
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GREEN = "#174638"
GREEN_2 = "#2F7657"
MINT = "#E3F2E8"
CREAM = "#F7F7F1"
TEXT = "#1E3028"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
:root {{ --green:{GREEN}; --green2:{GREEN_2}; --mint:{MINT}; --cream:{CREAM}; }}
.stApp {{ background: #fbfcfa; color:{TEXT}; }}
[data-testid="stHeader"] {{ background: rgba(251,252,250,.92); }}
.block-container {{ max-width: 1120px; padding-top: 1.3rem; padding-bottom: 3rem; }}
.hero {{
 background: linear-gradient(135deg, #174638 0%, #245b47 100%);
 color: white; border-radius: 24px; padding: 34px 30px; margin-bottom: 22px;
 box-shadow: 0 10px 28px rgba(23,70,56,.12);
}}
.hero .eyebrow {{ color:#b9e4c9; font-size:.9rem; font-weight:700; letter-spacing:.04em; }}
.hero h1 {{ color:white; font-size: clamp(2rem, 5vw, 3rem); margin:.4rem 0; font-weight:800; }}
.hero p {{ color:#e2eee7; font-size:1.05rem; margin:0; }}
.section-title {{ font-size:1.25rem; font-weight:800; color:{GREEN}; margin: 1.2rem 0 .8rem; }}
.soft-card {{
 background:white; border:1px solid #e5ebe5; border-radius:20px; padding:22px;
 box-shadow: 0 5px 18px rgba(20,50,35,.045); margin-bottom:14px;
}}
.feature-card {{
 background:white; border:1px solid #e5ebe5; border-radius:18px; padding:18px;
 min-height:130px; height:100%;
}}
.feature-icon {{ font-size:1.6rem; }}
.feature-card h3 {{ font-size:1.03rem; color:{GREEN}; margin:.45rem 0 .25rem; }}
.feature-card p {{ color:#66756c; font-size:.9rem; margin:0; }}
div.stButton > button {{
 border-radius: 999px; border:0; background:{GREEN_2}; color:white;
 font-weight:700; padding:.55rem 1.2rem;
}}
div.stButton > button:hover {{ background:{GREEN}; color:white; }}
[data-testid="stFileUploader"] section {{
 border: 1.5px dashed #a7bcae; border-radius:18px; background:#fcfdfb;
}}
div[data-testid="stMetric"] {{
 background:#f3f8f3; border:1px solid #e0ece2; border-radius:16px; padding:14px;
}}
.small-muted {{ color:#718078; font-size:.88rem; }}
.footer {{ color:#839087; font-size:.82rem; text-align:center; padding:24px 0 6px; }}
</style>
""", unsafe_allow_html=True)

# Language and navigation
if "lang" not in st.session_state:
    st.session_state.lang = "EN"
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Home"

lang = st.session_state.lang
AR = lang == "AR"

def t(en, ar):
    return ar if AR else en

def set_page(page):
    st.session_state.page = page

def get_model():
    try:
        import tensorflow as tf
        model_path = Path("PlantCare_AI_finetuned_best.keras")
        if not model_path.exists():
            return None, "Model file not found."
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

def get_classes():
    p = Path("PlantCare_AI_classes.json")
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Supports common JSON layouts
        for key in ("class_names", "classes", "class_indices"):
            if key in data:
                value = data[key]
                if isinstance(value, dict):
                    return [k for k, v in sorted(value.items(), key=lambda x: x[1])]
                return value
    return None

def predict(image):
    model, error = get_model()
    if model is None:
        return None, None, error
    classes = get_classes()
    if not classes:
        return None, None, "Class names file is missing or has an unsupported format."
    import tensorflow as tf
    img = image.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    scores = np.asarray(model.predict(arr, verbose=0))[0]
    if len(scores) != len(classes):
        return None, None, f"Model outputs {len(scores)} classes, but class file contains {len(classes)}."
    idx = int(np.argmax(scores))
    return classes[idx], float(scores[idx]), None

def pretty_label(label):
    return label.replace("___", " — ").replace("_", " ")

def care_text(label):
    # Safe general guidance; avoid unsupported pesticide prescriptions.
    if label and "healthy" in label.lower():
        return t(
            "The model classified this image as a healthy-looking leaf. Continue regular watering appropriate to the plant, provide suitable light, and monitor new growth.",
            "صنّف النموذج الصورة على أنها ورقة تبدو سليمة. استمري في الري المناسب للنبات، وتوفير الإضاءة الملائمة، ومتابعة النمو الجديد."
        )
    return t(
        "This is an AI screening result, not a confirmed diagnosis. Isolate the affected plant if practical, avoid wetting foliage unnecessarily, remove heavily damaged leaves with clean tools, and check the plant's growing conditions. Confirm the cause with a local agricultural specialist before using any treatment.",
        "هذه نتيجة فحص مبدئي بالذكاء الاصطناعي وليست تشخيصًا مؤكدًا. يُفضّل عزل النبات المصاب إن أمكن، وتجنب تبليل الأوراق دون حاجة، وإزالة الأوراق شديدة التلف بأداة نظيفة، ومراجعة ظروف الزراعة. تأكدي من السبب مع مختص زراعي قبل استخدام أي علاج."
    )

# Top bar
top1, top2 = st.columns([5, 1])
with top1:
    st.markdown("### 🌱 PlantCare AI")
with top2:
    if st.button("العربية" if not AR else "English", use_container_width=True):
        st.session_state.lang = "AR" if not AR else "EN"
        st.rerun()

pages = [
    ("Home", t("Home", "الرئيسية")),
    ("Plant Scanner", t("Plant Scanner", "فحص النبات")),
    ("Analysis Result", t("Analysis Result", "نتيجة التحليل")),
    ("Care Guide", t("Care Guide", "دليل العناية")),
    ("Scan History", t("Scan History", "سجل الفحوصات")),
    ("About", t("About", "عن التطبيق")),
]
nav_cols = st.columns(6)
for col, (key, label) in zip(nav_cols, pages):
    with col:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            set_page(key)
            st.rerun()

page = st.session_state.page

if page == "Home":
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">✦ AI POWERED PLANT CARE</div>
      <h1>{t("Keep your plants healthy.", "حافظي على صحة نباتاتك.")}</h1>
      <p>{t("Discover what's happening with your plant.", "اكتشفي ما يحدث لنباتك من خلال صورة للأوراق.")}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{t("Analyze your plant", "افحصي نباتك")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write(t("Upload or capture a leaf photo to begin.", "ارفعي صورة للورقة أو التقطي صورة بالكاميرا لبدء الفحص."))
    photo = st.file_uploader(t("Choose a leaf image", "اختاري صورة الورقة"), type=["jpg", "jpeg", "png"], key="home_upload")
    if photo:
        image = Image.open(photo).convert("RGB")
        st.image(image, use_container_width=True)
        if st.button(t("Analyze image", "تحليل الصورة"), key="home_analyze"):
            with st.spinner(t("Analyzing image...", "جارٍ تحليل الصورة...")):
                label, score, error = predict(image)
            if error:
                st.error(t("The model could not be loaded: ", "تعذر تشغيل النموذج: ") + error)
            else:
                record = {"label": label, "score": score, "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "image": image}
                st.session_state.latest = record
                st.session_state.history.insert(0, record)
                set_page("Analysis Result")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{t("Your plant care companion", "رفيقك للعناية بالنباتات")}</div>', unsafe_allow_html=True)
    feature_data = [
        ("📷", t("Smart plant scan", "فحص ذكي للنبات"), t("Upload a leaf image for AI analysis.", "ارفعي صورة ورقة للحصول على تحليل مبدئي.")),
        ("🌿", t("Care guidance", "إرشادات العناية"), t("Get practical general care tips.", "احصلي على إرشادات عامة للعناية.")),
        ("🕘", t("Scan history", "سجل الفحوصات"), t("Review your previous scans.", "راجعي الفحوصات السابقة.")),
    ]
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, feature_data):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

elif page == "Plant Scanner":
    st.markdown(f'<div class="section-title">{t("Plant Scanner", "فحص النبات")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write(t("Upload a clear photo of one plant leaf. Use good natural lighting and keep the leaf in focus.", "ارفعي صورة واضحة لورقة نبات واحدة. استخدمي إضاءة جيدة وتأكدي أن الورقة واضحة وفي بؤرة الكاميرا."))
    photo = st.file_uploader(t("Upload leaf photo", "رفع صورة الورقة"), type=["jpg", "jpeg", "png"], key="scanner_upload")
    if photo:
        image = Image.open(photo).convert("RGB")
        st.image(image, caption=t("Image preview", "معاينة الصورة"), use_container_width=True)
        if st.button(t("Start AI analysis", "ابدئي تحليل الذكاء الاصطناعي")):
            with st.spinner(t("Analyzing...", "جارٍ التحليل...")):
                label, score, error = predict(image)
            if error:
                st.error(error)
            else:
                record = {"label": label, "score": score, "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "image": image}
                st.session_state.latest = record
                st.session_state.history.insert(0, record)
                set_page("Analysis Result")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Analysis Result":
    st.markdown(f'<div class="section-title">{t("Analysis Result", "نتيجة التحليل")}</div>', unsafe_allow_html=True)
    result = st.session_state.get("latest")
    if not result:
        st.info(t("No analysis yet. Upload a leaf image first.", "لا توجد نتيجة حتى الآن. ارفعي صورة الورقة أولًا."))
        if st.button(t("Go to Plant Scanner", "الانتقال إلى فحص النبات")):
            set_page("Plant Scanner")
            st.rerun()
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(result["image"], use_container_width=True)
        with c2:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.caption(t("Predicted class", "الفئة المتوقعة"))
            st.subheader(pretty_label(result["label"]))
            st.metric(t("Model score", "درجة النموذج"), f"{result['score']*100:.1f}%")
            st.caption(t("This score is not a guarantee of diagnostic accuracy.", "هذه النسبة لا تضمن صحة التشخيص."))
            st.markdown('</div>', unsafe_allow_html=True)
        st.warning(t("AI screening only: results may be less reliable for real-world photos that differ from the PlantVillage dataset. Confirm important decisions with an agricultural specialist.", "الفحص بالذكاء الاصطناعي استرشادي فقط؛ قد تقل دقته مع الصور الواقعية المختلفة عن صور مجموعة PlantVillage. استشيري مختصًا زراعيًا قبل اتخاذ قرارات علاجية مهمة."))
        st.markdown(f'<div class="section-title">{t("Care guidance", "إرشادات العناية")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="soft-card">{care_text(result["label"])}</div>', unsafe_allow_html=True)

elif page == "Care Guide":
    st.markdown(f'<div class="section-title">{t("Care Guide", "دليل العناية")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write(t("General plant-care checklist", "قائمة إرشادات عامة للعناية بالنباتات"))
    tips = [
        t("Check soil moisture before watering.", "افحصي رطوبة التربة قبل الري."),
        t("Provide light suitable for the plant species.", "وفّري إضاءة مناسبة لنوع النبات."),
        t("Ensure pots have adequate drainage.", "تأكدي من وجود تصريف جيد للمياه."),
        t("Inspect both sides of leaves regularly.", "افحصي وجهي الأوراق بانتظام."),
        t("Clean tools between plants to reduce spread.", "نظّفي الأدوات بين النباتات لتقليل انتقال المشكلات."),
    ]
    for tip in tips:
        st.markdown(f"- {tip}")
    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.get("latest"):
        st.markdown(f'<div class="section-title">{t("Guidance for your latest scan", "إرشادات آخر فحص")}</div>', unsafe_allow_html=True)
        st.info(care_text(st.session_state.latest["label"]))

elif page == "Scan History":
    st.markdown(f'<div class="section-title">{t("Scan History", "سجل الفحوصات")}</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info(t("Your scans will appear here after analysis.", "ستظهر الفحوصات هنا بعد إجراء التحليل."))
    else:
        for i, item in enumerate(st.session_state.history):
            with st.container():
                a, b = st.columns([1, 3])
                with a:
                    st.image(item["image"], use_container_width=True)
                with b:
                    st.markdown(f"**{pretty_label(item['label'])}**")
                    st.write(f"{item['score']*100:.1f}% · {item['time']}")
        st.caption(t("History is stored for this session only and clears when the app session resets.", "يُحفظ السجل خلال الجلسة الحالية فقط، وقد يُمسح عند إعادة تشغيل الجلسة."))

elif page == "About":
    st.markdown(f'<div class="section-title">{t("About PlantCare AI", "عن PlantCare AI")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write(t("PlantCare AI is an educational plant-leaf image screening project powered by a fine-tuned MobileNetV2 model.", "PlantCare AI مشروع تعليمي لفحص صور أوراق النباتات باستخدام نموذج MobileNetV2 بعد تحسين تدريبه."))
    st.write(t("Dataset: PlantVillage · Supported classes: 38 · Input size: 224 × 224 RGB", "مجموعة البيانات: PlantVillage · عدد الفئات المدعومة: 38 · حجم الصورة: 224 × 224 RGB"))
    st.write(t("Fine-tuned model test accuracy: 87.80%. This is an evaluation result on the project's test set, not a promise of the same performance on every uploaded photo.", "دقة النموذج بعد التحسين على مجموعة الاختبار: 87.80%. هذه نتيجة تقييم على مجموعة اختبار المشروع، وليست ضمانًا لتحقيق النسبة نفسها مع كل صورة يتم رفعها."))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{t("Important limitations", "حدود مهمة")}</div>', unsafe_allow_html=True)
    st.warning(t("The model supports only its 38 trained classes. It may misclassify unfamiliar plants, poor-quality images, or symptoms not represented in the dataset. It does not replace expert agricultural diagnosis.", "يدعم النموذج الفئات الـ38 التي تدرب عليها فقط. وقد يخطئ مع النباتات غير المعروفة أو الصور منخفضة الجودة أو الأعراض غير الموجودة في البيانات. ولا يغني عن التشخيص لدى مختص زراعي."))

st.markdown('<div class="footer">🌱 PlantCare AI · Your Smart Plant Health Companion</div>', unsafe_allow_html=True)
