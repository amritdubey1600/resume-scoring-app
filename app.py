import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Extract resume text from PDF
def extract_resume_text(uploaded_file):
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp_resume.pdf")
    pages = loader.load()
    return " ".join([page.page_content for page in pages])

# Step 2: Vectorize and compute cosine similarity
def get_similarity_score(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

# Streamlit UI
st.title("🧠 Resume Scoring App")
st.write("Upload your resume and paste a job description to see how well they match.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if uploaded_file and job_desc:
    if st.button("Get Score"):
        with st.spinner("Analyzing..."):
            resume_text = extract_resume_text(uploaded_file)
            score = get_similarity_score(resume_text, job_desc)

        st.success(f"✅ Resume Match Score: **{score}/100**")

        if score > 75:
            st.info("🟢 Strong match! Great job aligning your resume.")
        elif score > 50:
            st.warning("🟡 Decent match. You may want to tweak a few sections.")
        else:
            st.error("🔴 Low match. Consider tailoring your resume more to this job.")
else:
    st.button("Get Score", disabled=True)
