# 🧠 Universal RAG Q&A App

A lightweight, fast, and private **Retrieval-Augmented Generation (RAG)** application that allows you to upload documents and ask intelligent questions using a local Large Language Model (LLM). Built with ❤️ for secure, local use with no data sent to the cloud.

---

## 🚀 Features

- 📁 Upload and parse CSV, PDF, DOCX, and TXT documents
- 🔍 Ask natural language questions about your documents
- 🧠 Uses **local LLMs** (like LLaMA 3.2) for fast and private inference
- ⚡ Fast document embedding via `nomic-embed-text`
- 🔎 Vector search powered by `ChromaDB`
- 🎨 Beautiful and responsive UI
- 🖥️ Fully offline and private (no internet connection required)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Ollama
- **LLM**: LLaMA3.2 (or your choice of local LLM)
- **Embedding Model**: `nomic-embed-text`
- **Database**: ChromaDB

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/abid4850/universal-rag-qa.git
cd universal-rag-qa

# Install dependencies
pip install -r requirements.txt

# Optional: Start Ollama and pull LLM model
ollama run llama3:latest

# Run the app
streamlit run app.py

## 📂 How to Use  
---

1. Launch the app with `streamlit run app.py`  
2. Upload your document (PDF, Word, CSV, or TXT)  
3. Ask questions about your uploaded content  
4. Get quick, context-rich answers from your local model!

---

## 📄 Supported Formats  
---

- `.pdf`  
- `.docx`  
- `.txt`  
- `.csv`

---

## 🔐 Privacy First  
---

All document processing and model inference happens **locally**. No data leaves your machine, making this app perfect for sensitive or private information workflows.

---

## 🤖 Requirements  
---

- Python 3.8+  
- pip  
- Ollama (for running LLMs locally)  
- Streamlit  
- ChromaDB  
- SentenceTransformers (for embeddings)

---

## 📜 License  
---

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## ✨ Credits  
---

Made with ❤️ using:

- [Ollama](https://ollama.com)  
- [ChromaDB](https://www.trychroma.com/)  
- [Streamlit](https://streamlit.io)

---

## 📬 Contact  
---

Feel free to reach out with feedback or suggestions:  
**Abid Hussain** | [GitHub](https://github.com/abid4850)
