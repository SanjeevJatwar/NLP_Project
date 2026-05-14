# RAG Evaluation Pipeline

This is a Retrieval-Augmented Generation (RAG) evaluation tool built with **Streamlit**, designed to assess the accuracy, factual consistency, and grounding of LLM responses against a curated medical dataset using **ChromaDB**.

## Features & Metrics Evaluated
When a query is entered, the pipeline retrieves relevant documents from the database, generates an answer via **Groq (Llama 3)**, and evaluates the result across three distinct metrics:

1. **Similarity Score (25%)**: Cosine similarity between the user's query and the retrieved database chunks to ensure relevance.
2. **NLI Entailment Score (45%)**: A Natural Language Inference model (`cross-encoder/nli-deberta-v3-small`) checks if the generated answer's factual claims are strictly entailed (supported) by the retrieved context.
3. **Entropy / Variance Score (30%)**: Uses Reverse Prompting (converting the answer back into a question and re-answering it) to check for self-consistency and eliminate hallucinations.

## Project Structure
- `app.py`: The main Streamlit web application containing the evaluation pipeline.
- `chroma_db/`: The pre-computed vector database containing the embedded medical dataset. *(Note: This eliminates the need to run data ingestion dynamically, saving massive amounts of memory and time on the server!)*
- `requirements.txt`: Required pip dependencies.
- `.env`: (Local Only) Contains secure API keys.

## 🚀 How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API Key:**
   Ensure you have a `.env` file in this directory with your Groq API Key:
   ```env
   GROQ_API_KEY="your_api_key_here"
   ```

3. **Start the Application:**
   Run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
Thank You !!!
---
*Built with Streamlit, LangChain, and Groq.*
