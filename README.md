# HR Policy Retrieval Assistant

**Designed for zero hallucination. Built with retrieval, not generation.**

---

## Overview

This project is a retrieval-first HR policy assistant that answers employee queries using internal policy documents.

Instead of using LLMs, the system uses **TF-IDF + cosine similarity** to return **accurate, source-backed answers**.

> If the answer exists in the document, generate nothing — retrieve it.

---

## Problem

Most AI assistants:

* hallucinate answers
* lack source traceability
* are unreliable for compliance use-cases

For HR policies:

> correctness > creativity

---

## Solution

A deterministic retrieval system that:

* searches across HR policy PDFs
* ranks relevant sections
* returns exact answers with sources

---

## Architecture

```text
PDFs → Text Extraction → Chunking → TF-IDF Vectorization → Similarity Search → Top-K Results
```

---

## Features

* ✅ Zero hallucination (no generation layer)
* ✅ Source traceability (document + page)
* ✅ Deterministic outputs
* ✅ Lightweight & fast
* ✅ No external APIs

---

## Tech Stack

* Python
* Scikit-learn (TF-IDF)
* Streamlit
* PyPDF2

---

## How to Run

```bash
pip install -r requirements.txt
python src/ingestion.py
streamlit run app.py
```

---

## Example

**Query:**

> cab reimbursement in mumbai

**Answer:**

* ₹800 per trip
* Applicable only for late-night work beyond 9:30 PM
* Source: Travel Policy 2024 (Page 1)

---

## Design Decision

LLMs were intentionally not used.

For structured, high-stakes domains like HR:

* retrieval > generation
* accuracy > fluency

---

## Author

**Anjana B**
AI Product | ABSLI
