# -*- coding: utf-8 -*-
"""tamminimongodp.ipynb"""

import streamlit as st
from pymongo import MongoClient
from datetime import datetime

# ----------------- Database Connection -----------------
uri = "mongodb+srv://tammeni25:mentalhealth255@tamminicluster.nunk6nw.mongodb.net/?retryWrites=true&w=majority&authSource=admin"
client = MongoClient(uri)
db = client["tammini_db"]
users_col = db["users"]
responses_col = db["responses"]

# ----------------- Page Config -----------------
st.set_page_config(page_title="منصة طَمّني", layout="centered", page_icon="🧠")

# ----------------- Landing Page -----------------
def show_landing_page():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
        <style>
        html, body, .stApp {
            background-color: #e6f7ff;
            font-family: 'Cairo', sans-serif;
            direction: rtl;
        }
        .landing-container {
            text-align: center;
            padding: 40px;
            background-color: #b3e0ff;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        h1 {
            color: #005b99;
            font-size: 48px;
            margin-bottom: 10px;
        }
        h3 {
            color: #222;
            font-size: 24px;
            margin-bottom: 20px;
        }
        </style>

        <div class='landing-container'>
            <h1>طَمّني</h1>
            <h3>منصة تقييم الصحة النفسية باستخدام الذكاء الاصطناعي</h3>
            <img src='https://cdn-icons-png.flaticon.com/512/4320/4320337.png' width='130' alt='brain icon'/>
        </div>
    """, unsafe_allow_html=True)

    if st.button("تسجيل الدخول / إنشاء حساب"):
        st.session_state.page = "auth"

# ----------------- Auth -----------------
def signup():
    st.subheader("🔐 تسجيل حساب جديد")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("تسجيل"):
        existing_user = users_col.find_one({"username": username})
        if existing_user:
            st.warning("هذا المستخدم مسجل بالفعل. يتم عرض التقرير الأول:")
            existing_response = responses_col.find_one({"username": username}, sort=[("timestamp", 1)])
            if existing_response:
                st.markdown("### 📂 التقرير الأول للمستخدم:")
                st.write(f"👤 الجنس: {existing_response['gender']}")
                st.write(f"📅 العمر: {existing_response['age']}")
                for i in range(1, 7):
                    st.write(f"س{i}: {existing_response.get(f'q{i}', '')}")
                if "result" in existing_response:
                    st.success(f"✅ النتيجة: {existing_response['result']}")
                else:
                    st.info("📌 لم يتم تحليل النتيجة بعد.")
            else:
                st.info("🔍 لا توجد ردود سابقة.")
        else:
            users_col.insert_one({"username": username, "password": password})
            st.success("✅ تم التسجيل بنجاح! يمكنك الآن تسجيل الدخول.")

def login():
    st.subheader("🔑 تسجيل الدخول")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("دخول"):
        user = users_col.find_one({"username": username, "password": password})
        if user:
            st.session_state['user'] = username
            st.success("مرحباً بك، تم تسجيل الدخول.")

            if st.button("📜 عرض سجل المستخدم"):
                history = responses_col.find({"username": username}).sort("timestamp", -1)
                for i, resp in enumerate(history, 1):
                    st.markdown(f"---\n### 📝 المحاولة رقم {i}:")
                    st.write(f"⏰ التاريخ: {resp['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"👤 الجنس: {resp['gender']}")
                    st.write(f"📅 العمر: {resp['age']}")
                    for qn in range(1, 7):
                        st.write(f"س{qn}: {resp.get(f'q{qn}', '')}")
                    if "result" in resp:
                        st.success(f"✅ النتيجة: {resp['result']}")
                    else:
                        st.info("📌 لم يتم تحليل النتيجة بعد.")
        else:
            st.error("بيانات الدخول غير صحيحة.")

# ----------------- Questionnaire -----------------
def questionnaire():
    st.subheader("📝 التقييم النفسي")
    gender = st.radio("ما هو جنسك؟", ["ذكر", "أنثى"])
    age = st.radio("ما هي فئتك العمرية؟", ["18-29", "30-39", "40-49", "50+"])

    questions = {
        1: "س1: ھل تجد نفسك تعاني من التفكير المفرط أو القلق الزائد تجاه مختلف الأمور الحياتية المحيطة بك، سواء كانت متعلقة بالعمل، الدراسة، المنزل، أو غيرها من الجوانب اليومية؟ اعط أمثلة على بعض من هذه الأمور وكيف يؤثر التفكير والقلق بها على أفكارك وسلوكك خلال اليوم.",
        2: "س2: ھل تواجه صعوبة في السيطرة على أفكارك القلقة أو التحكم في مستوى القلق الذي تشعر به، بحيث تشعر أن الأمر خارج عن إرادتك أو أنه مستمر على نحو يرهقك؟ اجعل إجابتك تفصيلية بحيث توضح كيف يكون خارج عن ارادتك أو إلى أي مدى يرهقك.",
        3: "س3: ھل يترافق مع التفكير المفرط أو القلق المستمر ثلاثة أعراض أو أكثر من الأعراض التالية: الشعور بعدم الارتياح أو بضغط نفسي كبير، الإحساس بالتعب والإرهاق بسهولة، صعوبة واضحة في التركيز، الشعور بالعصبية الزائدة، شد عضلي مزمن، اضطرابات في النوم، وغيرها؟ اذكر كل عرض تعاني منه وهل يؤثر على مهامك اليومية مثل العمل أو الدراسة أو حياتك الاجتماعية؟ وكيف يؤثر عليك بشكل يومي؟",
        4: "س4: ھل مررت بفترة استمرت أسبوعين أو أكثر كنت تعاني خلالها من خمسة أعراض أو أكثر مما يلي، مع ضرورة وجود عرض المزاج المكتئب أو فقدان الشغف والاهتمام؟ اذكر الأعراض التي عانيت منها بالتفصيل و كيف أثرت عليك؟",
        5: "س5: ھل أدت الأعراض التي مررت بها إلى شعورك بضيق نفسي شديد أو إلى تعطيل واضح لقدرتك على أداء مهامك اليومية، سواء في حياتك الاجتماعية، الوظيفية، أو الشخصية؟ كيف لاحظت تأثير ذلك عليك وعلى تفاعلاتك مع من حولك؟",
        6: "س6: ھل هذه الأعراض التي عانيت منها لم تكن ناتجة عن تأثير أي مواد مخدرة، أدوية معينة، أو بسبب حالة مرضية عضوية أخرى قد تكون أثرت على سلوكك أو مشاعرك خلال تلك الفترة؟"
    }

    answers = {}
    for i in range(1, 7):
        answers[f"q{i}"] = st.text_area(questions[i])

    if st.button("إرسال التقييم"):
        if any(ans.strip() == "" for ans in answers.values()):
            st.error("❗ الرجاء الإجابة على جميع الأسئلة.")
        elif any(any(char.isascii() and char.isalpha() for char in ans) for ans in answers.values()):
            st.error("❗ الرجاء عدم استخدام أحرف إنجليزية في الإجابات.")
        else:
            user = st.session_state.get('user')
            if user:
                responses_col.insert_one({
                    "username": user,
                    "gender": gender,
                    "age": age,
                    **answers,
                    "timestamp": datetime.now()
                })
                st.success("✅ تم حفظ الإجابات.")
            else:
                st.error("⚠️ يرجى تسجيل الدخول أولاً.")

# ----------------- Navigation -----------------
if 'page' not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    show_landing_page()
    st.stop()

if 'user' not in st.session_state:
    page = st.radio("اختر الصفحة", ["تسجيل الدخول", "تسجيل جديد"], horizontal=True)
    if page == "تسجيل الدخول":
        login()
    else:
        signup()
    st.stop()
else:
    questionnaire()
