import streamlit as st
from fastai.learner import load_learner
from PIL import Image

# Load model (make sure 'screws_model.pkl' is in the repo!)
learn = load_learner("screws_model.pkl")

categories = ('m4', 'm5')

def classify_img(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

st.title("Screw Classifier 🔩")
st.write("Upload an image of a screw (m4 or m5) to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    with st.spinner("Classifying..."):
        preds = classify_img(img)
    
    st.subheader("Predictions")
    st.json(preds)
