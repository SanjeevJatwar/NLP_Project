import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import numpy as np
import re
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

# Langchain imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Safely load the key from Streamlit Secrets or Local .env file
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

PERSIST_DIR = os.path.join(SCRIPT_DIR, "chroma_db")         # <-- Path where ChromaDB was saved in train.py

st.set_page_config(page_title="RAG Evaluation Pipeline", layout="wide")

# ==========================================
# CACHE MODELS SO THEY DON'T RELOAD ON EVERY UI INTERACTION
# ==========================================
@st.cache_resource
def load_models():
    # Sentence embedding model for Similarity Score (Query vs Chunks)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # NLI model for entailment check
    # Using device=0 since you mentioned having a GPU with CUDA 11.8
    nli_model = hf_pipeline(
        "text-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=0  
    )
    
    # Same embedding model used during storage
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Load existing vector store
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
        collection_name="medicine_collection"
    )
    
    # Convert to retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    return embedder, nli_model, retriever

# Stop early if the API key is not configured
if not GROQ_API_KEY or GROQ_API_KEY == "your_api_key_here":
    st.error("⚠️ Please configure your Groq API Key in your `.env` file or Streamlit Secrets to continue.")
    st.stop()

# Initialize Client and Models
groq_client = Groq(api_key=GROQ_API_KEY)

try:
    embedder, nli_model, retriever = load_models()
except Exception as e:
    st.error(f"Error loading models or ChromaDB. Did you run `train.py` first?\n\n{e}")
    st.stop()

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================
def retrieve_chunks(query):
    docs = retriever.invoke(query)
    chunks = []
    for doc in docs:
        parts = doc.page_content.split(" | ")
        # Return only Name, Composition, Uses, and Side Effects
        if len(parts) >= 4:
            chunks.append(f"{parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} |")
        else:
            chunks.append(doc.page_content)
    return chunks

def get_context(chunks):
    return " ".join(chunks)

def generate_answer(query, context):
    if isinstance(context, list):
        context = "\n\n".join([str(c) for c in context])
    if not context or context.strip() == "":
        return "No relevant information found in the database."
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Answer ONLY based on the provided context. If the answer is not present, say 'Not found in context.'"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer clearly and concisely."}
            ],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_reverse_prompt(original_query, answer):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a reverse prompt generator.\nGiven an original question and its answer, extract the key fact\nand rephrase it as a closed yes/no verification question.\n\nExamples:\nOriginal: What medicine treats fever?\nAnswer: Paracetamol treats fever.\nReverse: Is Paracetamol used to treat fever?\n\nReturn ONLY the reverse question."},
                {"role": "user", "content": f"Original:\n{original_query}\n\nAnswer:\n{answer}\n\nReverse question:"}
            ],
            temperature=0.0,
            max_tokens=128
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating reverse prompt: {str(e)}"

def answer_reverse_prompt(reverse_query, context):
    if isinstance(context, list):
        context = "\n\n".join([str(c) for c in context])
    if not context or context.strip() == "":
        return "No relevant information found in the database."
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer ONLY based on the provided context. If unsure, say 'Not found in context.'"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{reverse_query}\n\nAnswer clearly."}
            ],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def get_tfidf_distributions(texts):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts).toarray()
    dists = []
    for row in matrix:
        total = row.sum()
        dists.append(row / total if total > 0 else row)
    return dists

def cross_entropy(p, q, eps=1e-10):
    return float(-np.sum(p * np.log(q + eps)))

def mean_variance(distributions):
    matrix = np.array(distributions)
    return float(np.mean(np.var(matrix, axis=0)))

def compute_entropy_variance_score(original_answer, reverse_answer):
    try:
        dists = get_tfidf_distributions([original_answer, reverse_answer])
        p, q = dists[0], dists[1]
        ce_pq = cross_entropy(p, q)
        ce_qp = cross_entropy(q, p)
        mean_ce = (ce_pq + ce_qp) / 2
        variance = mean_variance(dists)
        ce_norm = max(0.0, 1.0 - mean_ce / 20.0)
        var_norm = max(0.0, 1.0 - variance / 0.01)
        entropy_score = round((ce_norm + var_norm) / 2, 4)
        return {
            "ce_orig_to_rev": round(ce_pq, 4),
            "ce_rev_to_orig": round(ce_qp, 4),
            "mean_ce": round(mean_ce, 4),
            "variance": round(variance, 6),
            "entropy_score": entropy_score
        }
    except Exception:
        return {
            "ce_orig_to_rev": 0.0, "ce_rev_to_orig": 0.0, "mean_ce": 0.0, "variance": 0.0, "entropy_score": 0.0
        }

def check_entailment(premise, hypothesis):
    premise = str(premise)
    hypothesis = str(hypothesis)
    result = nli_model(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
    label = result[0]["label"].upper()
    score = round(float(result[0]["score"]), 4)
    if "ENTAIL" in label:
        label = "ENTAILMENT"
    elif "CONTRADICT" in label:
        label = "CONTRADICTION"
    else:
        label = "NEUTRAL"
    return label, score

def extract_claims(answer):
    if not answer or answer.strip() == "":
        return []
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Break the answer into short atomic factual claims.\nReturn ONLY a numbered list:\n\n1. Claim one.\n2. Claim two."},
                {"role": "user", "content": f"Answer:\n{answer}\n\nList all atomic claims:"}
            ],
            temperature=0.0,
            max_tokens=512
        )
        raw = response.choices[0].message.content.strip()
        claims = re.findall(r"(?:\d+\.\s|\-\s|\*\s)?(.+)", raw)
        claims = [c.strip() for c in claims if c.strip()]
        return claims if claims else [answer]
    except Exception:
        return [answer]

def compute_nli_score(answer, chunks):
    if not answer:
        return {"claims": [], "total_claims": 0, "entailed": 0, "nli_score": 0.0}
    processed_chunks = []
    for c in chunks:
        if hasattr(c, "page_content"):
            processed_chunks.append(c.page_content)
        else:
            processed_chunks.append(str(c))
    if not processed_chunks:
        return {"claims": [], "total_claims": 0, "entailed": 0, "nli_score": 0.0}
    claims = extract_claims(answer)
    results = []
    for claim in claims:
        best_label = "NEUTRAL"
        best_score = 0.0
        for chunk in processed_chunks:
            label, score = check_entailment(chunk, claim)
            if label == "ENTAILMENT" and score > best_score:
                best_label = "ENTAILMENT"
                best_score = score
            elif best_label != "ENTAILMENT" and score > best_score:
                best_label = label
                best_score = score
        results.append({
            "claim": claim,
            "verdict": best_label,
            "confidence": round(best_score, 4)
        })
    entailed = [r for r in results if r["verdict"] == "ENTAILMENT"]
    nli_score = round(len(entailed) / len(results), 4) if results else 0.0
    return {
        "claims": results,
        "total_claims": len(results),
        "entailed": len(entailed),
        "nli_score": nli_score
    }

def compute_similarity_score(query, chunks):
    if not chunks:
        return {"per_chunk_similarity": [], "similarity_score": 0.0}
    query_emb = embedder.encode([query])
    chunk_embs = embedder.encode(chunks)
    sims = cosine_similarity(query_emb, chunk_embs)[0]
    per_chunk = [round(float(s), 4) for s in sims]
    mean_sim = round(float(np.mean(sims)), 4)
    return {
        "per_chunk_similarity": per_chunk,
        "similarity_score": mean_sim
    }

WEIGHTS = {"similarity": 0.25, "nli": 0.45, "entropy": 0.30}

def aggregate_scores(similarity_score, nli_score, entropy_score):
    final = (
        WEIGHTS["similarity"] * similarity_score +
        WEIGHTS["nli"] * nli_score +
        WEIGHTS["entropy"] * entropy_score
    )
    return round(final, 4)

def get_verdict(score):
    if score >= 0.75:
        return "🟢 HIGH — Answer is well-grounded and consistent."
    elif score >= 0.50:
        return "🟡 MODERATE — Answer is partially supported; review carefully."
    else:
        return "🔴 LOW — Answer may contain hallucinations or is poorly supported."

def evaluate_rag(query):
    chunks = retrieve_chunks(query) or []
    context = get_context(chunks) if chunks else ""
    answer = generate_answer(query, context)
    reverse_q = generate_reverse_prompt(query, answer)
    reverse_a = answer_reverse_prompt(reverse_q, context)
    ev = compute_entropy_variance_score(answer, reverse_a)
    nli = compute_nli_score(answer, chunks)
    sim = compute_similarity_score(query, chunks)
    final_score = aggregate_scores(
        similarity_score=sim.get("similarity_score", 0.0),
        nli_score=nli.get("nli_score", 0.0),
        entropy_score=ev.get("entropy_score", 0.0)
    )
    return {
        "query": query,
        "answer": answer,
        "reverse_q": reverse_q,
        "reverse_a": reverse_a,
        "similarity": sim,
        "nli": nli,
        "entropy": ev,
        "final_score": final_score,
        "chunks": chunks
    }

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("RAG Evaluation Pipeline")
st.markdown("Evaluate answers retrieved using Similarity, NLI Entailment, and Entropy/Variance metrics.")

user_query = st.text_input("Enter your query:", "What medicine is used to treat fever?")

if st.button("Evaluate"):
    with st.spinner("Running evaluation pipeline..."):
        result = evaluate_rag(user_query)
        
    st.subheader("Results")
    st.write(f"**Query:** {result['query']}")
    st.write(f"**Answer:** {result['answer']}")
    
    st.subheader("Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Similarity Score", f"{result['similarity'].get('similarity_score', 0):.4f}")
    col2.metric("NLI Score", f"{result['nli'].get('nli_score', 0):.4f}")
    col3.metric("Entropy Score", f"{result['entropy'].get('entropy_score', 0):.4f}")
    col4.metric("Final Score", f"{result['final_score']:.4f}")
    
    st.write(f"**Verdict:** {get_verdict(result['final_score'])}")
    
    with st.expander("Detailed Pipeline Output"):
        st.write(f"**Reverse Q:** {result['reverse_q']}")
        st.write(f"**Reverse A:** {result['reverse_a']}")
        st.write("**NLI Claim Breakdown:**")
        for i, c in enumerate(result['nli'].get('claims', []), 1):
            st.write(f"{i}. [{c.get('verdict', 'NEUTRAL')} {c.get('confidence', 0):.2f}] {c.get('claim', '')}")
            
        st.write("**Per-Chunk Similarity:**")
        for i, s in enumerate(result['similarity'].get("per_chunk_similarity", []), 1):
            st.write(f"Chunk {i}: {s:.4f}")
        
        st.write("**Retrieved Chunks:**")
        for i, chunk in enumerate(result['chunks'], 1):
            st.text_area(f"Chunk {i}", chunk, height=100)
