# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# モデル読み込み
model = load_model("face_model.h5")
class_names = ['Asian', 'Black', 'Indian', 'Others', 'White']


import streamlit as st

# ## はmarkdownにおけるheader
st.markdown("## 【画像から人種を推測するアプリ】")

st.markdown("""
<span style='color:gray;'>投稿日：2025年6月6日</span>  
&nbsp;  
<span style='background-color:#e0f7fa; color:#00796b; padding:4px 8px; border-radius:6px;'>Python</span>
<span style='background-color:#e8f5e9; color:#388e3c; padding:4px 8px; border-radius:6px;'>初心者</span>
<span style='background-color:#fff3e0; color:#ef6c00; padding:4px 8px; border-radius:6px;'>機械学習</span>
<span style='background-color:#fce4ec; color:#c2185b; padding:4px 8px; border-radius:6px;'>画像認識</span>
<span style='background-color:#ede7f6; color:#512da8; padding:4px 8px; border-radius:6px;'>Streamlit</span>
""", unsafe_allow_html=True)

st.markdown( # """は複数行
    """
    <div style="background-color:#fff3cd; padding:15px; border-radius:10px;">
        このアプリは顔画像から人種の特徴を推定するものであり、<br>
        <strong>差別的意図は含んでおらず、</strong><br>
        あくまでエンタメおよび技術デモンストレーションを目的としております。<br>
        ご理解のうえ、軽い気持ちでご利用いただければ幸いです。
    </div>
    """,
    unsafe_allow_html=True
)

st.write("") # 空行を追加
st.write("")

st.markdown("### <span style='color:black;'> 仕様</span> ", unsafe_allow_html=True)
st.write("""
    - Asian
    - Black
    - Indian
    - White
    - Others
    """)
st.write("と出力される。")

st.write("") # 空行を追加
st.write("")

uploaded_file = st.file_uploader("顔画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None: # uploadされたら
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="アップロード画像", use_column_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    st.write(f"### 🧠💡 推測された人種: **{pred_class}**")
