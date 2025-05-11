# app.py
import streamlit as st
from utils import process_text, get_top_chunks, generate_answer_with_rag, extract_ontology



# Must be FIRST Streamlit command
st.set_page_config(page_title="📖Ayush Welcome's You")


st.title("  📚Drop Your Books Here📚") 
st.header("     And Ask Any Question") 
# Upload or URL input
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
book_url = st.text_input("Or paste a URL to a .txt file (e.g., Project Gutenberg):")

text = None
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
elif book_url:
    import requests
    try:
        r = requests.get(book_url)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        st.error(f"❌ Failed to load book: {e}")

if text:
    st.success("✅ Book loaded. Processing for further analysis...")
    chunks, index = process_text(text)
    ontology = extract_ontology(text)
    with st.sidebar:
        st.markdown("## 🧠 Top Ontologies are below:")
        for label, items in ontology.items():
            #st.markdown(f"**{label}** ({len(items)}):")
            st.markdown(f"**{label}**:")
            st.markdown(", ".join(items))
            
    query = st.text_input("Ask a question from the book:")
    if query:
        top_chunks = get_top_chunks(query, index, chunks)
        answer = generate_answer_with_rag(query, top_chunks)
        st.markdown("### ✅ Answer:")
        st.write(answer)
else:
    st.info("Please upload a `.txt` file or paste a book URL.")
