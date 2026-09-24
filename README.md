# Corporate X-Ray

## One-Agent Autonomous UK Company Due-Diligence & Corporate Intelligence System

Corporate X-Ray automates repetitive UK company research using one AI agent connected to official Companies House data and documentary evidence retrieval.

## Core Architecture

- 1 AI Agent
- 8 Tools
- Companies House REST API
- Hugging Face Qwen2.5-7B-Instruct
- Hybrid BM25 + semantic retrieval
- BGE reranking
- Filing-document retrieval
- PDF extraction
- Corporate event analysis
- Streamlit interface

## Eight Tools

1. company_search
2. company_profile
3. company_officers
4. company_pscs
5. company_filings
6. company_charges
7. company_insolvency
8. search_company_evidence

## Problem Statement

UK company due-diligence often requires manually identifying the correct legal entity, checking the company profile, reviewing officers, reviewing PSC information, inspecting filing history, checking charges and insolvency information, opening filing documents, searching for relevant evidence, and assembling findings into a report.

Corporate X-Ray coordinates this workflow through one AI agent and specialized deterministic tools.

## Manual Work Reduced

- Repeated Companies House navigation
- Manual company identification
- Manual officer inspection
- Manual PSC inspection
- Manual filing-history review
- Manual charge review
- Manual insolvency lookup
- Manually opening filing documents
- Manually searching filing PDFs
- Manually collecting document and page references
- Manually assembling the investigation report

## RAG Pipeline

Companies House filing history
-> Document metadata
-> PDF retrieval
-> PyMuPDF extraction
-> Page-level chunks
-> BM25 + Semantic Search
-> Reciprocal Rank Fusion
-> BGE Reranker
-> Documentary Evidence

## Technology Stack

- Python
- Hugging Face Transformers
- Qwen2.5-7B-Instruct
- smolagents
- Companies House REST API
- PyMuPDF
- BM25
- BGE embeddings
- BGE reranker
- Pandas / NumPy
- Streamlit

## Local Setup

Create a virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create .env from .env.example and add your credentials.

Never commit .env.

## Run

streamlit run app.py

## Project Structure

Corporate-XRay/
|-- app.py
|-- corporate_xray_agentic_rag.py
|-- requirements.txt
|-- .gitignore
|-- .env.example
|-- README.md
`-- Corporate_XRay_Agentic_RAG.ipynb

## Portfolio Skills Demonstrated

- Agentic AI
- Tool orchestration
- Retrieval Augmented Generation
- Hybrid search
- Reranking
- Document intelligence
- API integration
- Evidence-grounded generation
- Corporate-data analysis
- GPU-aware inference
- Streamlit development

## Important Limitation

Corporate X-Ray is an information-retrieval and intelligence-support system. It is not a substitute for legal advice, accounting advice, or regulated compliance review.