# app.py

import streamlit as st
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

VECT_DIR = "tfidf_index"

# ------------------ LOAD INDEX ------------------

with open(os.path.join(VECT_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(VECT_DIR, "tfidf_matrix.pkl"), "rb") as f:
    tfidf_matrix = pickle.load(f)

with open(os.path.join(VECT_DIR, "metadata.pkl"), "rb") as f:
    store = pickle.load(f)

documents = store["documents"]
metadata = store["metadata"]

# ------------------ SEARCH ------------------

def search(query, top_k=5):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]

    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append({
            "text": documents[i],
            "source": metadata[i]["source"],
            "page": metadata[i]["page"],
            "score": float(scores[i])
        })

    return results


# ------------------ STRUCTURED ANSWER ------------------

def generate_structured_answer(query, results):
    combined = " ".join(r["text"] for r in results).lower()

    answer = {
        "Duration": None,
        "Eligibility": [],
        "Conditions": [],
        "Documents Required": [],
        "Category": None
    }

    if "sick" in query.lower():
        answer["Category"] = "Leave Policy"
        answer["Duration"] = "8 days per calendar year"

        if "mental" in combined:
            answer["Conditions"].append(
                "Out of 8 sick leave days, 2 may be used as Mental Health Days without medical certificate."
            )

        if "certificate" in combined:
            answer["Documents Required"].append(
                "Medical certificate required for illness exceeding 2 consecutive days."
            )

    if "cab" in query.lower() or "reimbursement" in query.lower():
        answer["Category"] = "Travel & Reimbursement"
        answer["Duration"] = "₹800 per trip"
        answer["Conditions"].append(
            "Applicable only for late-night project-related work beyond 9:30 PM via Uber/Ola."
        )

    return answer


# ------------------ UI ------------------

st.set_page_config(page_title="HR Policy Assistant", layout="wide")
st.title("🤖 HR Policy Assistant")
st.caption("Policy-grounded internal HR assistant with structured answers.")

query = st.text_input("Ask HR:")

if query:
    results = search(query)
    structured = generate_structured_answer(query, results)

    st.success("Suggested Answer (Structured)")

    col1, col2 = st.columns(2)

    with col1:
        if structured["Duration"]:
            st.markdown("### 🗓 Duration")
            st.write(structured["Duration"])

        if structured["Eligibility"]:
            st.markdown("### 👤 Eligibility")
            for e in structured["Eligibility"]:
                st.write("•", e)

    with col2:
        if structured["Conditions"]:
            st.markdown("### ⚠️ Conditions")
            for c in structured["Conditions"]:
                st.write("•", c)

        if structured["Documents Required"]:
            st.markdown("### 📄 Documents Required")
            for d in structured["Documents Required"]:
                st.write("•", d)

    if structured["Category"]:
        st.markdown(f"📌 **Category:** {structured['Category']}")

    st.markdown("### 📚 Sources")
    for r in results:
        st.write(
            f"- {r['source']} (Page {r['page']}) — score {round(r['score'], 2)}"
        )
