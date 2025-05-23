📚 **RAG-based QA System for Any Book**

This project is a **Retrieval-Augmented Generation (RAG)** system built by **Ayush Shrivastava** for case study. It allows users to upload or link any `.txt` book (e.g., from Project Gutenberg), ask natural language questions, and get answers generated from context using models like **FLAN-T5**.

---

🔍 Features

- ✅ Load any `.txt` book (uploaded file or URL)
- ✅ Automatically chunk and embed the book using `sentence-transformers`
- ✅ Fast semantic search using FAISS
- ✅ Query answering powered by **FLAN-T5** (local, no API needed, Few more model are already setted up and can be tested as per the desired result.)
- ✅ Additionally Open AI API has been setted up If you anyone wants to use, can replace the API key with the existing dummy API key and uncomment the function below it.
- ✅ Sidebar ontology with **named entities** (people, places, organizations)
- ✅ Easily swappable book input and models

---

📦 How to Install and Use 


```bash
1. Clone this repo
bash:
git clone https://github.com/fit-coder-ayush/book-rag-qa.git
cd book-rag-qa

2. Install dependencies
bash:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

3. Run the APP
bash:
streamlit run app.py



	📁 File Structure
📦 book-rag-qa/
├── app.py                 # Streamlit UI
├── utils.py               # Core logic: chunking, embedding, QA
├── requirements.txt       # All dependencies
└── README.md              # Project overview



	📘 Example Use
✅ Paste a Project Gutenberg .txt URL ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt")
✅ Ask: “Who is Mr. Darcy?”
✅ Get the relevant answer.



	🧠 Ontology
In the sidebar, the app displays key named entities extracted from the book:
👤 People
🌍 Geopolitical Locations
🏛️ Organizations
These help users understand the character and setting landscape of the text.



	🛠️ Customization
🔁 Model: Easily switch from flan-t5-base to flan-t5-large or GPT by editing utils.py
📚 Input: Replace the .txt file or URL with any new book
🤖 API version (optional): You can integrate GPT-3.5/4 via OpenAI if needed and already having an API key



Thanks for reading,
In case of any queries or library dependencies please revert back to me on below contacts
Mail: shrivastava.ayush181297@gmail.com
Phone: +91 70497 94984