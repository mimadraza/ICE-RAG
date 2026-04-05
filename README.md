# 🛡️ ICE-RAG — Know Your Rights Assistant

**An AI-powered chatbot that helps individuals and communities understand their legal rights during ICE (Immigration and Customs Enforcement) raids and encounters.**

> *This tool is built to inform, not to replace legal counsel. If you or someone you know is in immediate danger or has been detained, contact an immigration attorney immediately.*

---

![ICE-RAG App Preview](./preview.png)
*The ICE-RAG interface — a knowledge oracle built to help communities stand informed against ICE raids.*

---

## 📖 What Is This?

ICE-RAG is a **Retrieval-Augmented Generation (RAG)** application that allows anyone to ask plain-language questions about their constitutional and legal rights when faced with an ICE raid or encounter. Instead of searching through dense legal documents, you can simply ask:

- *"Do I have to open the door for ICE officers?"*
- *"What rights do I have if ICE comes to my workplace?"*
- *"Can ICE enter my home without a warrant?"*
- *"What should I do if I am detained?"*

The system retrieves relevant legal information from a curated knowledge base and generates accurate, accessible answers grounded in that information.

---

## ⚙️ How It Works

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────────────┐     ┌──────────────────┐
│  Embedding  │────▶│   Vector Database Search  │────▶│  LLM Generation  │
│   Model     │     │  (Know Your Rights Docs)  │     │  (Grounded Answer│
└─────────────┘     └──────────────────────────┘     └──────────────────┘
```

1. **Ingestion** — Legal documents, ACLU guides, immigration rights resources, and government policy documents are chunked and embedded into a vector store.
2. **Retrieval** — When you ask a question, the most relevant document chunks are retrieved based on semantic similarity.
3. **Generation** — A language model uses the retrieved context to generate a clear, grounded answer — minimizing hallucination.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key (or compatible LLM provider)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/mimadraza/ICE-RAG.git
cd ICE-RAG

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
```

### Running the App

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:8501` (if using Streamlit) or the URL shown in your terminal.

---

## 📚 Knowledge Base

The RAG system is built on a curated set of trusted sources, including:

- ACLU "Know Your Rights" guides for immigrants
- National Immigration Law Center (NILC) resources
- ILRC (Immigrant Legal Resource Center) red cards and materials
- U.S. Constitutional rights documentation (4th, 5th, and 14th Amendments)
- State-specific sanctuary city and immigration policies

To add or update documents in the knowledge base, place `.pdf` or `.txt` files in the `/docs` directory and re-run the ingestion script:

```bash
python ingest.py
```

---

## 🗂️ Project Structure

```
ICE-RAG/
├── app.py              # Main application entry point
├── ingest.py           # Document ingestion and embedding pipeline
├── rag.py              # Core RAG logic (retrieval + generation)
├── docs/               # Source documents for the knowledge base
├── vectorstore/        # Persisted vector database
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variable template
```

---

## 🤝 Contributing

Community contributions are welcome — especially:

- **New documents**: Legal guides, multilingual resources, state-specific rights info
- **Translations**: Making this tool accessible in Spanish, Haitian Creole, and other languages
- **Testing**: Helping verify the accuracy of generated responses against known legal facts

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-spanish-docs`)
3. Commit your changes and open a Pull Request

---

## ⚠️ Disclaimer

This tool is for **informational purposes only** and does not constitute legal advice. Immigration law is complex and situation-dependent. Always consult a qualified immigration attorney for guidance specific to your circumstances.

**Emergency Resources:**
- 🔴 **RAICES Emergency Hotline**: 1-888-RAICES-1
- 🔴 **ACLU Immigration Rights**: aclu.org/know-your-rights/immigrants-rights
- 🔴 **Immigration Advocates Network**: immigrationadvocates.org

---

## 📄 License

This project is open source. See [LICENSE](./LICENSE) for details.

---

*Built to protect the rights of everyone — regardless of status.*
