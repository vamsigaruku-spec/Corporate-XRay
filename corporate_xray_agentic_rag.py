import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pymupdf
import requests
import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from smolagents import Model, ToolCallingAgent, tool
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    get_tool_json_schema,
)

PROJECT_DIR = Path(__file__).resolve().parent

CH_API_URL = "https://api.company-information.service.gov.uk"
DOC_API_URL = "https://document-api.company-information.service.gov.uk"

MODEL_ID = os.getenv("CORPORATE_XRAY_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
EMBEDDING_MODEL_ID = os.getenv(
    "CORPORATE_XRAY_EMBEDDING_MODEL_ID",
    "BAAI/bge-small-en-v1.5",
)
RERANKER_MODEL_ID = os.getenv(
    "CORPORATE_XRAY_RERANKER_MODEL_ID",
    "BAAI/bge-reranker-base",
)

WORKFLOW = [
    "company_search",
    "company_profile",
    "company_officers",
    "company_pscs",
    "company_filings",
    "company_charges",
    "company_insolvency",
    "search_company_evidence",
    "final_answer",
]
TOOL_NAMES = WORKFLOW[:-1]

MAX_EVIDENCE_ATTEMPTS = 3

def load_dotenv_file():
    """Load simple KEY=VALUE pairs from a local .env file."""
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value

load_dotenv_file()

def _get_config_value(name, default=""):
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        value = str(st.secrets.get(name, default)).strip()
    except Exception:
        value = str(default).strip()
    return value

COMPANIES_HOUSE_API_KEY = _get_config_value("COMPANIES_HOUSE_API_KEY")
HUGGINGFACE_TOKEN = (
    _get_config_value("HUGGINGFACEHUB_API_TOKEN")
    or _get_config_value("HF_TOKEN")
)
OPENROUTER_API_KEY = _get_config_value("OPENROUTER_API_KEY") or _get_config_value("LLM_API_KEY")
OPENROUTER_BASE_URL = _get_config_value(
    "LLM_BASE_URL",
    "https://openrouter.ai/api/v1",
)
OPENROUTER_PRIMARY_MODEL = _get_config_value(
    "LLM_MODEL_NAME",
    "qwen/qwen-2.5-72b-instruct",
)
OPENROUTER_FALLBACK_MODEL = _get_config_value(
    "LLM_FALLBACK_MODEL_NAME",
    "qwen/qwen-2.5-7b-instruct",
)
OPENROUTER_SITE_URL = _get_config_value("OPENROUTER_SITE_URL")
OPENROUTER_SITE_NAME = _get_config_value(
    "OPENROUTER_SITE_NAME",
    "Corporate X-Ray",
)
DEPLOYMENT_MODE = _get_config_value(
    "CORPORATE_XRAY_DEPLOYMENT",
    "local",
).lower()

def _session():
    defaults = {
        "corporate_xray_state": {
            "current_stage": "company_search",
            "company_search_completed": False,
            "company_profile_completed": False,
            "officers_completed": False,
            "pscs_completed": False,
            "filings_completed": False,
            "charges_completed": False,
            "insolvency_completed": False,
            "evidence_completed": False,
            "selected_company_name": None,
            "selected_company_number": None,
            "investigation_question": "",
            "default_evidence_query": "",
            "evidence_attempts": 0,
            "evidence_queries": [],
            "evidence_sufficient": False,
            "started_at": None,
        },
        "corporate_xray_data": {
            "company_search": None,
            "company_profile": None,
            "officers": None,
            "pscs": None,
            "filings": None,
            "charges": None,
            "insolvency": None,
        },
        "corporate_xray_evidence": [],
        "rag_state": {
            "company_number": None,
            "chunks": [],
            "bm25": None,
            "document_embeddings": None,
        },
        "_agent": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value
    return st.session_state

def _xray_state():
    return _session()["corporate_xray_state"]

def _xray_data():
    return _session()["corporate_xray_data"]

def _evidence():
    return _session()["corporate_xray_evidence"]

def _rag_state():
    return _session()["rag_state"]

def require_companies_house_api_key():
    key = _get_config_value("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise RuntimeError(
            "COMPANIES_HOUSE_API_KEY is missing. Add it to Streamlit Secrets or local .env."
        )
    return key

def check_backend_health():
    """Return (healthy, message) without throwing."""
    try:
        if DEPLOYMENT_MODE not in {"local", "cloud"}:
            return False, "CORPORATE_XRAY_DEPLOYMENT must be 'local' or 'cloud'."
        if not require_companies_house_api_key():
            return False, "COMPANIES_HOUSE_API_KEY is missing."
        if DEPLOYMENT_MODE == "cloud" and not (
            _get_config_value("OPENROUTER_API_KEY")
            or _get_config_value("LLM_API_KEY")
        ):
            return False, "OPENROUTER_API_KEY (or LLM_API_KEY) is missing for cloud deployment."
        return True, "Backend configuration is valid."
    except Exception as exc:
        return False, str(exc)


# ============================================================
# 7. RESET STATE
# ============================================================

def reset_corporate_xray_state():
    """Reset the current user's investigation state."""
    state = _session()

    state["corporate_xray_state"] = {
        "current_stage": "company_search",
        "company_search_completed": False,
        "company_profile_completed": False,
        "officers_completed": False,
        "pscs_completed": False,
        "filings_completed": False,
        "charges_completed": False,
        "insolvency_completed": False,
        "evidence_completed": False,
        "selected_company_name": None,
        "selected_company_number": None,
        "investigation_question": "",
        "default_evidence_query": "",
        "evidence_attempts": 0,
        "evidence_queries": [],
        "evidence_sufficient": False,
        "started_at": None,
    }

    state["corporate_xray_data"] = {
        "company_search": None,
        "company_profile": None,
        "officers": None,
        "pscs": None,
        "filings": None,
        "charges": None,
        "insolvency": None,
    }

    state["corporate_xray_evidence"] = []

    state["rag_state"] = {
        "company_number": None,
        "chunks": [],
        "bm25": None,
        "document_embeddings": None,
    }

    state["_agent"] = None
# 8. WORKFLOW HELPERS
# ============================================================

def advance_stage(completed_stage):
    """Advance the guarded structured workflow. Evidence remains agent-controlled."""
    if completed_stage == "search_company_evidence":
        if _xray_state()["evidence_attempts"] >= MAX_EVIDENCE_ATTEMPTS:
            _xray_state()["evidence_completed"] = True
            _xray_state()["current_stage"] = "final_answer"
        else:
            _xray_state()["current_stage"] = "search_company_evidence"
        return

    index = WORKFLOW.index(completed_stage)
    if index + 1 < len(WORKFLOW):
        _xray_state()["current_stage"] = WORKFLOW[index + 1]
    else:
        _xray_state()["current_stage"] = "final_answer"


def current_company_number():
    """
    Return the selected company number.
    """

    number = (
        _xray_state().get(
            "selected_company_number"
        )
    )

    if not number:

        raise RuntimeError(
            "No company has been selected."
        )

    return number


# ============================================================
# 9. COMPANIES HOUSE API
# ============================================================

def companies_house_get(
    endpoint,
    params=None,
    document_api=False,
):
    """
    Perform authenticated GET request.
    """

    base_url = (
        DOC_API_URL
        if document_api
        else CH_API_URL
    )

    response = requests.get(
        f"{base_url}{endpoint}",

        auth=(
            require_companies_house_api_key(),
            "",
        ),

        params=params,

        timeout=60,
    )

    if response.status_code == 404:

        return None

    if not response.ok:

        raise RuntimeError(
            "Companies House request failed "
            f"({response.status_code}): "
            f"{response.text[:500]}"
        )

    return response.json()


# ============================================================
# 10. SEARCH COMPANY
# ============================================================

def search_company_api(
    query,
    limit=10,
):
    """
    Search Companies House.
    """

    data = companies_house_get(
        "/search/companies",

        params={
            "q":
                query,

            "items_per_page":
                limit,
        },
    )

    if not data:
        return []

    results = []

    for item in data.get(
        "items",
        [],
    ):

        address = (
            item.get(
                "address"
            )
            or {}
        )

        results.append({

            "company_name":
                item.get(
                    "title"
                ),

            "company_number":
                item.get(
                    "company_number"
                ),

            "company_status":
                item.get(
                    "company_status"
                ),

            "company_type":
                item.get(
                    "company_type"
                ),

            "date_of_creation":
                item.get(
                    "date_of_creation"
                ),

            "address_snippet":
                address.get(
                    "address_line_1"
                ),
        })

    return results


# ============================================================
# 11. COMPANY PROFILE
# ============================================================

def get_company_profile(
    company_number,
):
    """
    Retrieve official company profile.
    """

    data = companies_house_get(
        f"/company/{company_number}"
    )

    if data is None:

        return {
            "error":
                (
                    f"Company "
                    f"{company_number} "
                    "was not found."
                )
        }

    accounts = (
        data.get(
            "accounts"
        )
        or {}
    )

    last_accounts = (
        accounts.get(
            "last_accounts"
        )
        or {}
    )

    next_accounts = (
        accounts.get(
            "next_accounts"
        )
        or {}
    )

    confirmation = (
        data.get(
            "confirmation_statement"
        )
        or {}
    )

    address = (
        data.get(
            "registered_office_address"
        )
        or {}
    )

    return {

        "company_name":
            data.get(
                "company_name"
            ),

        "company_number":
            data.get(
                "company_number"
            ),

        "company_status":
            data.get(
                "company_status"
            ),

        "company_type":
            data.get(
                "type"
            ),

        "date_of_creation":
            data.get(
                "date_of_creation"
            ),

        "date_of_cessation":
            data.get(
                "date_of_cessation"
            ),

        "jurisdiction":
            data.get(
                "jurisdiction"
            ),

        "sic_codes":
            data.get(
                "sic_codes",
                [],
            ),

        "registered_office": {

            "address_line_1":
                address.get(
                    "address_line_1"
                ),

            "locality":
                address.get(
                    "locality"
                ),

            "postal_code":
                address.get(
                    "postal_code"
                ),

            "country":
                address.get(
                    "country"
                ),
        },

        "accounts": {

            "last_period_end":
                last_accounts.get(
                    "period_end_on"
                ),

            "last_accounts_type":
                last_accounts.get(
                    "type"
                ),

            "next_accounts_due":
                next_accounts.get(
                    "due_on"
                ),

            "accounts_overdue":
                next_accounts.get(
                    "overdue"
                ),
        },

        "confirmation_statement": {

            "last_made_up_to":
                confirmation.get(
                    "last_made_up_to"
                ),

            "next_due":
                confirmation.get(
                    "next_due"
                ),

            "overdue":
                confirmation.get(
                    "overdue"
                ),
        },
    }


# ============================================================
# 12. OFFICERS
# ============================================================

def get_officers(
    company_number,
):
    """
    Retrieve company officers.
    """

    data = companies_house_get(
        f"/company/{company_number}/officers"
    )

    if not data:
        return []

    return [

        {

            "name":
                item.get(
                    "name"
                ),

            "role":
                item.get(
                    "officer_role"
                ),

            "appointed_on":
                item.get(
                    "appointed_on"
                ),

            "resigned_on":
                item.get(
                    "resigned_on"
                ),

            "nationality":
                item.get(
                    "nationality"
                ),

            "occupation":
                item.get(
                    "occupation"
                ),

            "country_of_residence":
                item.get(
                    "country_of_residence"
                ),
        }

        for item
        in data.get(
            "items",
            [],
        )
    ]


# ============================================================
# 13. PSC
# ============================================================

def get_pscs(
    company_number,
):
    """
    Retrieve persons with significant control.
    """

    data = companies_house_get(
        f"/company/{company_number}/"
        "persons-with-significant-control"
    )

    if not data:
        return []

    return [

        {

            "name":
                item.get(
                    "name"
                ),

            "kind":
                item.get(
                    "kind"
                ),

            "nature_of_control":
                item.get(
                    "natures_of_control",
                    [],
                ),

            "notified_on":
                item.get(
                    "notified_on"
                ),

            "ceased_on":
                item.get(
                    "ceased_on"
                ),
        }

        for item
        in data.get(
            "items",
            [],
        )
    ]


# ============================================================
# 14. FILING HISTORY
# ============================================================

def get_filing_history(
    company_number,
    limit=20,
):
    """
    Retrieve recent filing history.
    """

    data = companies_house_get(
        f"/company/{company_number}/filing-history",

        params={
            "items_per_page":
                limit,
        },
    )

    if not data:
        return []

    results = []

    for item in data.get(
        "items",
        [],
    ):

        links = (
            item.get(
                "links"
            )
            or {}
        )

        results.append({

            "date":
                item.get(
                    "date"
                ),

            "type":
                item.get(
                    "type"
                ),

            "description":
                item.get(
                    "description"
                ),

            "category":
                item.get(
                    "category"
                ),

            "action_date":
                item.get(
                    "action_date"
                ),

            "document_metadata":
                links.get(
                    "document_metadata"
                ),
        })

    return results


# ============================================================
# 15. CHARGES
# ============================================================

def get_charges(
    company_number,
):
    """
    Retrieve registered company charges.
    """

    data = companies_house_get(
        f"/company/{company_number}/charges"
    )

    if not data:
        return []

    return [

        {

            "charge_code":
                item.get(
                    "charge_code"
                ),

            "created_on":
                item.get(
                    "created_on"
                ),

            "delivered_on":
                item.get(
                    "delivered_on"
                ),

            "status":
                item.get(
                    "status"
                ),

            "satisfied_on":
                item.get(
                    "satisfied_on"
                ),

            "particulars":
                item.get(
                    "particulars"
                ),

            "classification":
                (
                    item.get(
                        "classification"
                    )
                    or {}
                ).get(
                    "description"
                ),
        }

        for item
        in data.get(
            "items",
            [],
        )
    ]


# ============================================================
# 16. INSOLVENCY
# ============================================================

def get_insolvency(
    company_number,
):
    """
    Retrieve insolvency information.
    """

    data = companies_house_get(
        f"/company/{company_number}/insolvency"
    )

    if data is None:

        return {
            "available":
                False,

            "cases":
                [],
        }

    return {

        "available":
            True,

        "cases":
            data.get(
                "cases",
                [],
            ),
    }


# ============================================================
# 17. DOCUMENT ID
# ============================================================

def extract_document_id(
    document_url,
):
    """
    Extract Companies House document ID.
    """

    path = urlparse(
        document_url
    ).path

    parts = [
        part
        for part
        in path.split("/")
        if part
    ]

    if "document" not in parts:

        raise ValueError(
            "Unexpected document metadata URL."
        )

    index = parts.index(
        "document"
    )

    if index + 1 >= len(parts):

        raise ValueError(
            "Document ID missing."
        )

    return parts[index + 1]


# ============================================================
# 18. DOCUMENT DOWNLOAD
# ============================================================

def download_filing_document(
    document_id,
):
    """
    Download filing PDF.
    """

    response = requests.get(

        f"{DOC_API_URL}/"
        f"document/{document_id}/content",

        auth=(
            require_companies_house_api_key(),
            "",
        ),

        headers={
            "Accept":
                "application/pdf",
        },

        allow_redirects=True,

        timeout=60,
    )

    if not response.ok:

        raise RuntimeError(
            "Document download failed "
            f"({response.status_code})"
        )

    return response.content


# ============================================================
# 19. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_chunks(
    pdf_bytes,
    company_number,
    document_id,
    filing_date,
    category,
    chunk_words=420,
):
    """
    Extract page-aware chunks from PDF.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    chunks = []

    try:

        for page_index in range(
            document.page_count
        ):

            text = (
                document
                .load_page(
                    page_index
                )
                .get_text(
                    "text"
                )
                .strip()
            )

            if not text:
                continue

            words = text.split()

            for start in range(
                0,
                len(words),
                chunk_words,
            ):

                chunk = (
                    " ".join(
                        words[
                            start:
                            start
                            + chunk_words
                        ]
                    )
                    .strip()
                )

                if not chunk:
                    continue

                chunks.append({

                    "company_number":
                        company_number,

                    "document_id":
                        document_id,

                    "category":
                        category,

                    "filing_date":
                        filing_date,

                    "page":
                        page_index + 1,

                    "text":
                        chunk,
                })

    finally:

        document.close()

    return chunks


# ============================================================
# 20. TOKENIZATION
# ============================================================

def tokenize_text(
    text,
):
    """
    Basic BM25 tokenization.
    """

    cleaned = re.sub(
        r"[^a-zA-Z0-9]+",
        " ",
        str(text).lower(),
    )

    return cleaned.split()


# ============================================================
# 21. BUILD RAG INDEX
# ============================================================

def build_company_rag_index(
    company_number,
    max_documents=5,
):
    """
    Build compact hybrid RAG index.
    """

    filings = (
        _xray_data().get(
            "filings"
        )
        or []
    )

    candidates = [

        filing
        for filing
        in filings

        if filing.get(
            "document_metadata"
        )
    ]

    priority_types = {

        "AP01",
        "AP02",
        "AP03",
        "AP04",

        "CH01",
        "CH02",
        "CH03",
        "CH04",
        "CH05",

        "PSC01",
        "PSC02",
        "PSC03",
        "PSC04",
        "PSC05",

        "CS01",
    }

    candidates.sort(
        key=lambda item: (
            0 if item.get("type") in priority_types else 1,
            item.get("date") or "",
        ),
        reverse=False,
    )

    # Within each priority group, use the most recent filing first.
    candidates.sort(
        key=lambda item: item.get("date") or "",
        reverse=True,
    )

    candidates = candidates[
        :max_documents
    ]

    chunks = []

    for filing in candidates:

        try:

            document_id = (
                extract_document_id(
                    filing[
                        "document_metadata"
                    ]
                )
            )

            pdf_bytes = (
                download_filing_document(
                    document_id
                )
            )

            chunks.extend(
                extract_pdf_chunks(
                    pdf_bytes,

                    company_number,

                    document_id,

                    filing.get(
                        "date"
                    ),

                    filing.get(
                        "category"
                    ),
                )
            )

        except Exception:

            continue

    if not chunks:

        _rag_state().update({

            "company_number":
                company_number,

            "chunks":
                [],

            "bm25":
                None,

            "document_embeddings":
                None,
        })

        return 0

    tokenized = [

        tokenize_text(
            item["text"]
        )

        for item
        in chunks
    ]

    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(
        [item["text"] for item in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )

    _rag_state().update({

        "company_number":
            company_number,

        "chunks":
            chunks,

        "bm25":
            BM25Okapi(
                tokenized
            ),

        "document_embeddings":
            np.asarray(
                embeddings
            ),
    })

    return len(chunks)


def ensure_company_rag_index(
    company_number,
):
    """
    Reuse existing RAG index when possible.
    """

    if (
        _rag_state()[
            "company_number"
        ]
        == company_number

        and _rag_state()[
            "chunks"
        ]
    ):

        return len(
            _rag_state()[
                "chunks"
            ]
        )

    return build_company_rag_index(
        company_number
    )


# ============================================================
# 22. HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    candidate_k=12,
):
    """
    BM25 + semantic retrieval + RRF.
    """

    chunks = _rag_state()[
        "chunks"
    ]

    if not chunks:
        return []

    query_tokens = tokenize_text(
        query
    )

    bm25_scores = (
        _rag_state()[
            "bm25"
        ].get_scores(
            query_tokens
        )
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][
        :candidate_k
    ]

    query_embedding = get_embedding_model().encode(
        query,
        normalize_embeddings=True,
    )

    semantic_scores = np.dot(

        _rag_state()[
            "document_embeddings"
        ],

        query_embedding,
    )

    semantic_indices = np.argsort(
        semantic_scores
    )[::-1][
        :candidate_k
    ]

    rrf_scores = {}

    for rank, index in enumerate(
        bm25_indices
    ):

        key = int(index)

        rrf_scores[key] = (
            rrf_scores.get(
                key,
                0.0,
            )
            + 1.0 / (
                60
                + rank
                + 1
            )
        )

    for rank, index in enumerate(
        semantic_indices
    ):

        key = int(index)

        rrf_scores[key] = (
            rrf_scores.get(
                key,
                0.0,
            )
            + 1.0 / (
                60
                + rank
                + 1
            )
        )

    ranked_indices = sorted(

        rrf_scores.items(),

        key=lambda item:
            item[1],

        reverse=True,
    )

    candidates = []

    for index, score in ranked_indices:

        item = (
            _rag_state()[
                "chunks"
            ][index].copy()
        )

        item[
            "rrf_score"
        ] = float(
            score
        )

        text_lower = (
            item["text"].lower()
        )

        query_lower = (
            query.lower()
        )

        boost = 0.0

        if (
            "director"
            in query_lower

            or "appointment"
            in query_lower
        ):

            if (
                "appointment of director"
                in text_lower
            ):
                boost += 4.0

            if (
                "date of appointment"
                in text_lower
            ):
                boost += 3.0

        if (
            "ownership"
            in query_lower

            or "control"
            in query_lower

            or "psc"
            in query_lower
        ):

            if (
                "significant control"
                in text_lower
            ):
                boost += 4.0

        item[
            "combined_score"
        ] = (
            score + boost
        )

        candidates.append(
            item
        )

    candidates.sort(

        key=lambda item:
            item[
                "combined_score"
            ],

        reverse=True,
    )

    candidates = candidates[
        :candidate_k
    ]

    if not candidates:
        return []

    pairs = [

        [
            query,
            item["text"],
        ]

        for item
        in candidates
    ]

    scores = get_reranker().predict(pairs)

    for item, score in zip(
        candidates,
        scores,
    ):

        item[
            "reranker_score"
        ] = float(
            score
        )

    candidates.sort(

        key=lambda item:
            item[
                "reranker_score"
            ],

        reverse=True,
    )

    final_results = []

    seen = set()

    for item in candidates:

        key = (

            item.get(
                "document_id"
            ),

            item.get(
                "page"
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        final_results.append({

            "company_number":
                item.get(
                    "company_number"
                ),

            "document_id":
                item.get(
                    "document_id"
                ),

            "category":
                item.get(
                    "category"
                ),

            "filing_date":
                item.get(
                    "filing_date"
                ),

            "page":
                item.get(
                    "page"
                ),

            "score":
                item.get(
                    "reranker_score"
                ),

            "evidence":
                item.get(
                    "text",
                    "",
                )[
                    :2500
                ],
        })

        if len(
            final_results
        ) >= 3:

            break

    return final_results


# ============================================================
# 23. TOOL 1 — COMPANY SEARCH
# ============================================================
@tool
def company_search(query: str) -> dict:
    """
    Search Companies House for a UK company.

    Args:
        query: Company name or search term to search on Companies House.

    Returns:
        The selected company name and company number.
    """
    query = str(query).strip()

    if not query:
        return {
            "status": "error",
            "message": "Company name is required.",
        }

    # Use the actual Companies House API helper defined above.
    results = search_company_api(query, limit=10)

    if not results:
        return {
            "status": "not_found",
            "query": query,
            "message": f"No Companies House result found for '{query}'.",
        }

    query_normalized = query.upper()

    exact_matches = []
    other_matches = []

    for item in results:
        name = (
            item.get("company_name") or ""
        ).strip().upper()

        if name == query_normalized:
            exact_matches.append(item)
        else:
            other_matches.append(item)

    # Prefer exact active matches, then other active matches.
    exact_matches.sort(
        key=lambda item: item.get("company_status") != "active"
    )
    other_matches.sort(
        key=lambda item: item.get("company_status") != "active"
    )

    matches = exact_matches[:3] + other_matches[:2]

    selected = exact_matches[0] if exact_matches else matches[0]

    selected_name = selected.get("company_name")
    selected_number = selected.get("company_number")

    if not selected_number:
        return {
            "status": "error",
            "query": query,
            "message": "Companies House returned a result without a company number.",
            "matches": matches,
        }

    _xray_state()["selected_company_name"] = selected_name
    _xray_state()["selected_company_number"] = selected_number
    _xray_data()["company_search"] = {
        "query": query,
        "matches": matches,
        "selected_company_name": selected_name,
        "selected_company_number": selected_number,
    }
    _xray_state()["company_search_completed"] = True

    # Move to the next guarded stage.
    advance_stage("company_search")

    return {
        "status": "success",
        "query": query,
        "selected_company_name": selected_name,
        "selected_company_number": selected_number,
        "matches": matches,
        "next_stage": _xray_state()["current_stage"],
    }

# ============================================================
# 24. TOOL 2 — COMPANY PROFILE
# ============================================================

@tool
def company_profile() -> dict:
    """
    Retrieve the selected company's official profile.

    Returns:
        Company profile summary.
    """

    if (
        _xray_state()[
            "current_stage"
        ]
        != "company_profile"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                _xray_state()[
                    "current_stage"
                ],
        }

    company_number = (
        current_company_number()
    )

    result = get_company_profile(
        company_number
    )

    _xray_data()[
        "company_profile"
    ] = result

    _xray_state()[
        "company_profile_completed"
    ] = True

    advance_stage(
        "company_profile"
    )

    return {

        "status":
            "completed",

        "company_name":
            result.get(
                "company_name"
            ),

        "company_number":
            result.get(
                "company_number"
            ),

        "company_status":
            result.get(
                "company_status"
            ),

        "company_type":
            result.get(
                "company_type"
            ),

        "date_of_creation":
            result.get(
                "date_of_creation"
            ),

        "jurisdiction":
            result.get(
                "jurisdiction"
            ),

        "sic_codes":
            result.get(
                "sic_codes",
                [],
            ),
    }


# ============================================================
# 25. TOOL 3 — OFFICERS
# ============================================================

@tool
def company_officers() -> dict:
    """
    Retrieve officers of the selected company.

    Returns:
        Officer counts and sample records.
    """

    if (
        _xray_state()[
            "current_stage"
        ]
        != "company_officers"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                _xray_state()[
                    "current_stage"
                ],
        }

    result = get_officers(
        current_company_number()
    )

    _xray_data()[
        "officers"
    ] = result

    _xray_state()[
        "officers_completed"
    ] = True

    current = [

        item

        for item
        in result

        if not item.get(
            "resigned_on"
        )
    ]

    advance_stage(
        "company_officers"
    )

    return {

        "status":
            "completed",

        "total_officers":
            len(result),

        "current_officers":
            len(current),

        "current_officer_sample":
            current[:6],
    }


# ============================================================
# 26. TOOL 4 — PSC
# ============================================================

@tool
def company_pscs() -> dict:
    """
    Retrieve persons with significant control.

    Returns:
        PSC count and sample records.
    """

    if (
        _xray_state()[
            "current_stage"
        ]
        != "company_pscs"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                _xray_state()[
                    "current_stage"
                ],
        }

    result = get_pscs(
        current_company_number()
    )

    _xray_data()[
        "pscs"
    ] = result

    _xray_state()[
        "pscs_completed"
    ] = True

    advance_stage(
        "company_pscs"
    )

    return {

        "status":
            "completed",

        "total_pscs":
            len(result),

        "psc_sample":
            result[:6],
    }


# ============================================================
# 27. TOOL 5 — FILINGS
# ============================================================

@tool
def company_filings() -> dict:
    """
    Retrieve recent filing history.

    Returns:
        Filing count and recent filings.
    """

    if (
        _xray_state()[
            "current_stage"
        ]
        != "company_filings"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                _xray_state()[
                    "current_stage"
                ],
        }

    result = get_filing_history(
        current_company_number(),
        20,
    )

    _xray_data()[
        "filings"
    ] = result

    _xray_state()[
        "filings_completed"
    ] = True

    advance_stage(
        "company_filings"
    )

    return {

        "status":
            "completed",

        "filings_retrieved":
            len(result),

        "recent_filings":
            result[:8],
    }


# ============================================================
# 28. TOOL 6 — CHARGES
# ============================================================

@tool
def company_charges() -> dict:
    """
    Retrieve registered charges.

    Returns:
        Charge count and statuses.
    """

    if (
        _xray_state()[
            "current_stage"
        ]
        != "company_charges"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                _xray_state()[
                    "current_stage"
                ],
        }

    result = get_charges(
        current_company_number()
    )

    _xray_data()[
        "charges"
    ] = result

    _xray_state()[
        "charges_completed"
    ] = True

    status_counts = {}

    for item in result:

        status = (
            item.get(
                "status"
            )
            or "unknown"
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    advance_stage(
        "company_charges"
    )

    return {

        "status":
            "completed",

        "charges_retrieved":
            len(result),

        "status_counts":
            status_counts,

        "recent_charges":
            result[:6],
    }


# ============================================================
# 29. TOOL 7 — INSOLVENCY
# ============================================================

@tool
def company_insolvency() -> dict:
    """
    Retrieve insolvency information.

    Returns:
        Insolvency availability and case count.
    """

    if (
        _xray_state()[
            "current_stage"
        ]
        != "company_insolvency"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                _xray_state()[
                    "current_stage"
                ],
        }

    result = get_insolvency(
        current_company_number()
    )

    _xray_data()[
        "insolvency"
    ] = result

    _xray_state()[
        "insolvency_completed"
    ] = True

    advance_stage(
        "company_insolvency"
    )

    return {

        "status":
            "completed",

        "available":
            result.get(
                "available",
                False,
            ),

        "case_count":
            len(
                result.get(
                    "cases",
                    [],
                )
            ),
    }


# ============================================================
# 30. TOOL 8 — SEARCH DOCUMENTARY EVIDENCE
# ============================================================
@tool
def search_company_evidence(query: str) -> list:
    """
    Search official Companies House filing documents using hybrid RAG
    and reranking.

    Args:
        query: Natural-language question describing the documentary
            evidence that should be retrieved.

    Returns:
        A list of ranked documentary evidence records.
    """
    if _xray_state()["current_stage"] != "search_company_evidence":
        return [
            {
                "status": "blocked",
                "required_stage": _xray_state()["current_stage"],
            }
        ]

    query = str(query).strip()

    if not query:
        query = _xray_state().get(
            "default_evidence_query",
            "",
        ).strip()

    if not query:
        return [
            {
                "status": "error",
                "message": "A documentary evidence query is required.",
            }
        ]

    company_number = current_company_number()

    attempt = _xray_state().get("evidence_attempts", 0) + 1
    _xray_state()["evidence_attempts"] = attempt
    _xray_state()["evidence_queries"].append(query)

    try:
        chunk_count = ensure_company_rag_index(company_number)

        if not chunk_count:
            _xray_state()["evidence_completed"] = True
            _xray_state()["current_stage"] = "final_answer"
            return [
                {
                    "status": "no_evidence",
                    "company_number": company_number,
                    "message": (
                        "No documentary evidence is available "
                        "for the selected company."
                    ),
                    "attempt": attempt,
                }
            ]

        # hybrid_search already performs:
        # BM25 + dense retrieval + RRF + BGE reranking.
        results = hybrid_search(
            query,
            candidate_k=8,
        )

        normalized = [
            {
                "company_number": company_number,
                "document_id": item.get("document_id"),
                "category": item.get("category"),
                "filing_date": item.get("filing_date"),
                "page": item.get("page"),
                "score": item.get("reranker_score"),
                "evidence": item.get("evidence", ""),
            }
            for item in results
        ]

        # Keep all unique evidence across attempts.
        existing_keys = {
            (
                item.get("document_id"),
                item.get("page"),
                item.get("evidence"),
            )
            for item in _evidence()
        }

        for item in normalized:
            key = (
                item.get("document_id"),
                item.get("page"),
                item.get("evidence"),
            )
            if key not in existing_keys:
                _evidence().append(item)

        _xray_state()["evidence_sufficient"] = bool(normalized)
        _xray_state()["evidence_completed"] = bool(normalized)

        # The agent may either accept the evidence and finish or
        # request another search. Hard cap prevents infinite loops.
        if attempt >= MAX_EVIDENCE_ATTEMPTS:
            _xray_state()["evidence_completed"] = True
            _xray_state()["current_stage"] = "final_answer"

        return normalized

    except Exception as exc:
        if attempt >= MAX_EVIDENCE_ATTEMPTS:
            _xray_state()["evidence_completed"] = True
            _xray_state()["current_stage"] = "final_answer"

        return [
            {
                "status": "error",
                "company_number": company_number,
                "attempt": attempt,
                "message": f"Evidence retrieval failed: {exc}",
            }
        ]


# ============================================================
# 31. FINAL ANSWER VALIDATION
# ============================================================

def investigation_complete(final_answer, memory, agent=None):
    """Reject final answers until structured data and evidence are available."""
    required_flags = [
        "company_search_completed",
        "company_profile_completed",
        "officers_completed",
        "pscs_completed",
        "filings_completed",
        "charges_completed",
        "insolvency_completed",
    ]
    missing = [
        key for key in required_flags
        if not _xray_state().get(key, False)
    ]

    if missing:
        raise ValueError(
            "Investigation incomplete. Missing steps: " + ", ".join(missing)
        )

    if not _xray_state().get("selected_company_number"):
        raise ValueError(
            "FINAL ANSWER REJECTED: no verified company number is available."
        )

    if not _xray_state().get("evidence_attempts", 0):
        raise ValueError(
            "FINAL ANSWER REJECTED: documentary evidence search has not been attempted."
        )

    if not _evidence() and not _xray_state().get("evidence_completed"):
        raise ValueError(
            "FINAL ANSWER REJECTED: no documentary evidence was retrieved."
        )

    _xray_state()["evidence_completed"] = True
    return True


# ============================================================
# 32. LOCAL QWEN MODEL ADAPTER
# ============================================================

class LocalQwenModel(Model):
    """
    Qwen local adapter for smolagents.

    The model only receives the tool that is valid for the
    current workflow stage. This prevents the earlier:
      - wrong arguments
      - repeated tools
      - premature final answers
      - out-of-order calls
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens=96,
    ):

        super().__init__(

            model_id=
                MODEL_ID,

            max_new_tokens=
                max_new_tokens,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = (
            max_new_tokens
        )

    def _prepare_messages(
        self,
        messages,
    ):
        """
        Convert smolagents messages to Qwen chat format.
        """

        prepared = []

        for message in messages:

            if isinstance(
                message,
                ChatMessage,
            ):

                role = (
                    message.role.value
                )

                content = (
                    message.content
                )

            else:

                role = message.get(
                    "role",
                    "user",
                )

                content = message.get(
                    "content",
                    "",
                )

            if isinstance(
                content,
                list,
            ):

                parts = []

                for item in content:

                    if (
                        isinstance(
                            item,
                            dict,
                        )
                        and item.get(
                            "type"
                        ) == "text"
                    ):

                        parts.append(
                            str(
                                item.get(
                                    "text",
                                    "",
                                )
                            )
                        )

                content = "\n".join(
                    parts
                )

            if role == "tool-response":

                role = "user"

                content = (
                    "<tool_response>\n"
                    + str(content)[
                        -1800:
                    ]
                    + "\n</tool_response>"
                )

            elif role == "tool-call":

                role = "assistant"

            prepared.append({

                "role":
                    role,

                "content":
                    str(
                        content
                    )[
                        -1800:
                    ],
            })

        return prepared

    def _make_tool_call(
        self,
        generated_text,
    ):
        """Parse a Qwen tool call with safe stage-aware fallbacks."""
        stage = _xray_state()["current_stage"]

        match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            generated_text,
            re.DOTALL,
        )

        if match:
            try:
                payload = json.loads(match.group(1))
                if isinstance(payload, dict) and payload.get("name"):
                    name = payload["name"]
                    arguments = payload.get("arguments", {})
                    if not isinstance(arguments, dict):
                        arguments = {}

                    valid_names = {stage}
                    if stage == "search_company_evidence":
                        valid_names.add("final_answer")

                    if name in valid_names:
                        if name == "company_search" and not arguments.get("query"):
                            arguments["query"] = _xray_state().get("investigation_question", "")
                        if name == "search_company_evidence" and not arguments.get("query"):
                            arguments["query"] = _xray_state().get("default_evidence_query", "")
                        return {"name": name, "arguments": arguments}
            except Exception:
                pass

        if stage == "final_answer":
            return {
                "name": "final_answer",
                "arguments": {
                    "answer": generated_text.strip() or "Investigation completed.",
                },
            }

        if stage == "search_company_evidence":
            if _xray_state().get("evidence_attempts", 0) >= MAX_EVIDENCE_ATTEMPTS:
                return {
                    "name": "final_answer",
                    "arguments": {
                        "answer": generated_text.strip() or "Investigation completed.",
                    },
                }
            return {
                "name": "search_company_evidence",
                "arguments": {
                    "query": _xray_state().get("default_evidence_query", "")
                },
            }

        fallback_args = {}
        if stage == "company_search":
            fallback_args["query"] = _xray_state().get("investigation_question", "")

        return {"name": stage, "arguments": fallback_args}

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        """
        Generate a single agent action.
        """

        prepared = (
            self._prepare_messages(
                messages
            )
        )

        stage = (
            _xray_state()[
                "current_stage"
            ]
        )

        allowed_tools = []
        for candidate in (tools_to_call_from or []):
            if candidate.name == stage:
                allowed_tools.append(candidate)
            elif stage == "search_company_evidence" and candidate.name == "final_answer":
                allowed_tools.append(candidate)

        tool_schemas = [

            get_tool_json_schema(
                tool_obj
            )

            for tool_obj
            in allowed_tools
        ]

        inputs = (
            self.tokenizer
            .apply_chat_template(

                prepared,

                tools=
                    tool_schemas,

                add_generation_prompt=
                    True,

                tokenize=
                    True,

                return_dict=
                    True,

                return_tensors=
                    "pt",
            )
        )

        model_device = (
            self.model.device
        )

        inputs = {

            key:
                value.to(
                    model_device
                )

            for key, value
            in inputs.items()
        }

        prompt_length = (
            inputs[
                "input_ids"
            ].shape[-1]
        )

        max_new_tokens = int(

            kwargs.get(

                "max_new_tokens",

                self.max_new_tokens,
            )
        )

        import torch

        with torch.inference_mode():

            outputs = (
                self.model.generate(

                    **inputs,

                    max_new_tokens=
                        max_new_tokens,

                    do_sample=
                        False,

                    use_cache=
                        True,

                    pad_token_id=
                        self.tokenizer
                        .eos_token_id,
                )
            )

        generated_tokens = (
            outputs[0][
                prompt_length:
            ]
        )

        generated_text = (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=
                    True,
            )
            .strip()
        )

        payload = (
            self._make_tool_call(
                generated_text
            )
        )

        return ChatMessage(

            role=
                MessageRole.ASSISTANT,

            content=(
                "<tool_call>\n"
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n</tool_call>"
            ),
        )


# ============================================================
# 31. FINAL ANSWER VALIDATION
# ============================================================

def investigation_complete(final_answer, memory, agent=None):
    """Reject final answers until structured data and evidence are available."""
    required_flags = [
        "company_search_completed",
        "company_profile_completed",
        "officers_completed",
        "pscs_completed",
        "filings_completed",
        "charges_completed",
        "insolvency_completed",
    ]
    missing = [
        key for key in required_flags
        if not _xray_state().get(key, False)
    ]

    if missing:
        raise ValueError(
            "Investigation incomplete. Missing steps: " + ", ".join(missing)
        )

    if not _xray_state().get("selected_company_number"):
        raise ValueError(
            "FINAL ANSWER REJECTED: no verified company number is available."
        )

    if not _xray_state().get("evidence_attempts", 0):
        raise ValueError(
            "FINAL ANSWER REJECTED: documentary evidence search has not been attempted."
        )

    if not _evidence() and not _xray_state().get("evidence_completed"):
        raise ValueError(
            "FINAL ANSWER REJECTED: no documentary evidence was retrieved."
        )

    _xray_state()["evidence_completed"] = True
    return True


# ============================================================
# 32. LOCAL QWEN MODEL ADAPTER
# ============================================================

class LocalQwenModel(Model):
    """
    Qwen local adapter for smolagents.

    The model only receives the tool that is valid for the
    current workflow stage. This prevents the earlier:
      - wrong arguments
      - repeated tools
      - premature final answers
      - out-of-order calls
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens=96,
    ):

        super().__init__(

            model_id=
                MODEL_ID,

            max_new_tokens=
                max_new_tokens,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = (
            max_new_tokens
        )

    def _prepare_messages(
        self,
        messages,
    ):
        """
        Convert smolagents messages to Qwen chat format.
        """

        prepared = []

        for message in messages:

            if isinstance(
                message,
                ChatMessage,
            ):

                role = (
                    message.role.value
                )

                content = (
                    message.content
                )

            else:

                role = message.get(
                    "role",
                    "user",
                )

                content = message.get(
                    "content",
                    "",
                )

            if isinstance(
                content,
                list,
            ):

                parts = []

                for item in content:

                    if (
                        isinstance(
                            item,
                            dict,
                        )
                        and item.get(
                            "type"
                        ) == "text"
                    ):

                        parts.append(
                            str(
                                item.get(
                                    "text",
                                    "",
                                )
                            )
                        )

                content = "\n".join(
                    parts
                )

            if role == "tool-response":

                role = "user"

                content = (
                    "<tool_response>\n"
                    + str(content)[
                        -1800:
                    ]
                    + "\n</tool_response>"
                )

            elif role == "tool-call":

                role = "assistant"

            prepared.append({

                "role":
                    role,

                "content":
                    str(
                        content
                    )[
                        -1800:
                    ],
            })

        return prepared

    def _make_tool_call(
        self,
        generated_text,
    ):
        """Parse a Qwen tool call with safe stage-aware fallbacks."""
        stage = _xray_state()["current_stage"]

        match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            generated_text,
            re.DOTALL,
        )

        if match:
            try:
                payload = json.loads(match.group(1))
                if isinstance(payload, dict) and payload.get("name"):
                    name = payload["name"]
                    arguments = payload.get("arguments", {})
                    if not isinstance(arguments, dict):
                        arguments = {}

                    valid_names = {stage}
                    if stage == "search_company_evidence":
                        valid_names.add("final_answer")

                    if name in valid_names:
                        if name == "company_search" and not arguments.get("query"):
                            arguments["query"] = _xray_state().get("investigation_question", "")
                        if name == "search_company_evidence" and not arguments.get("query"):
                            arguments["query"] = _xray_state().get("default_evidence_query", "")
                        return {"name": name, "arguments": arguments}
            except Exception:
                pass

        if stage == "final_answer":
            return {
                "name": "final_answer",
                "arguments": {
                    "answer": generated_text.strip() or "Investigation completed.",
                },
            }

        if stage == "search_company_evidence":
            if _xray_state().get("evidence_attempts", 0) >= MAX_EVIDENCE_ATTEMPTS:
                return {
                    "name": "final_answer",
                    "arguments": {
                        "answer": generated_text.strip() or "Investigation completed.",
                    },
                }
            return {
                "name": "search_company_evidence",
                "arguments": {
                    "query": _xray_state().get("default_evidence_query", "")
                },
            }

        fallback_args = {}
        if stage == "company_search":
            fallback_args["query"] = _xray_state().get("investigation_question", "")

        return {"name": stage, "arguments": fallback_args}

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        """
        Generate a single agent action.
        """

        prepared = (
            self._prepare_messages(
                messages
            )
        )

        stage = (
            _xray_state()[
                "current_stage"
            ]
        )

        allowed_tools = []
        for candidate in (tools_to_call_from or []):
            if candidate.name == stage:
                allowed_tools.append(candidate)
            elif stage == "search_company_evidence" and candidate.name == "final_answer":
                allowed_tools.append(candidate)

        tool_schemas = [

            get_tool_json_schema(
                tool_obj
            )

            for tool_obj
            in allowed_tools
        ]

        inputs = (
            self.tokenizer
            .apply_chat_template(

                prepared,

                tools=
                    tool_schemas,

                add_generation_prompt=
                    True,

                tokenize=
                    True,

                return_dict=
                    True,

                return_tensors=
                    "pt",
            )
        )

        model_device = (
            self.model.device
        )

        inputs = {

            key:
                value.to(
                    model_device
                )

            for key, value
            in inputs.items()
        }

        prompt_length = (
            inputs[
                "input_ids"
            ].shape[-1]
        )

        max_new_tokens = int(

            kwargs.get(

                "max_new_tokens",

                self.max_new_tokens,
            )
        )

        import torch

        with torch.inference_mode():

            outputs = (
                self.model.generate(

                    **inputs,

                    max_new_tokens=
                        max_new_tokens,

                    do_sample=
                        False,

                    use_cache=
                        True,

                    pad_token_id=
                        self.tokenizer
                        .eos_token_id,
                )
            )

        generated_tokens = (
            outputs[0][
                prompt_length:
            ]
        )

        generated_text = (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=
                    True,
            )
            .strip()
        )

        payload = (
            self._make_tool_call(
                generated_text
            )
        )

        return ChatMessage(

            role=
                MessageRole.ASSISTANT,

            content=(
                "<tool_call>\n"
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n</tool_call>"
            ),
        )


# ============================================================

# ============================================================
# 33. LOCAL QWEN (LAZY + CACHED)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_qwen():
    """Load the local Qwen model only when local mode is selected."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HUGGINGFACE_TOKEN or None,
    )

    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=HUGGINGFACE_TOKEN or None,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quant_config,
                attn_implementation="sdpa",
            )
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=HUGGINGFACE_TOKEN or None,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HUGGINGFACE_TOKEN or None,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    return model, tokenizer


# ============================================================
# 34. OPENROUTER MODEL + ONE AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def _get_openrouter_client(api_key, base_url, site_url, site_name):
    """Create one shared stateless OpenRouter HTTP client."""
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-OpenRouter-Title"] = site_name

    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": 120.0,
        "max_retries": 0,
    }
    if headers:
        client_kwargs["default_headers"] = headers
    return OpenAI(**client_kwargs)


class OpenRouterModel(Model):
    """
    OpenRouter adapter implementing smolagents Model.generate().

    Primary model is Qwen2.5-72B-Instruct.
    Fallback model is Qwen2.5-7B-Instruct.
    OpenRouter itself performs provider-level routing/failover.
    """

    def __init__(
        self,
        api_key,
        base_url=OPENROUTER_BASE_URL,
        primary_model=OPENROUTER_PRIMARY_MODEL,
        fallback_model=OPENROUTER_FALLBACK_MODEL,
    ):
        super().__init__(model_id=primary_model)
        self.base_url = base_url
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.api_key = api_key
        self.client = _get_openrouter_client(
            api_key,
            base_url,
            OPENROUTER_SITE_URL,
            OPENROUTER_SITE_NAME,
        )

    @staticmethod
    def _status_code(exc):
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None)

    @staticmethod
    def _retry_after(exc):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return min(max(float(value), 0.5), 10.0)
        except (TypeError, ValueError):
            return 2.0

    @classmethod
    def _error_class(cls, exc):
        status = cls._status_code(exc)
        if status == 402:
            return "credit"
        if status == 429:
            return "rate_limit"
        if status is not None and 500 <= int(status) <= 599:
            return "server"
        if status in {408, 409, 425}:
            return "transient"
        if status in {401, 403}:
            return "auth"
        if status == 400:
            return "bad_request"
        text = str(exc).lower()
        if any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "connection reset",
                "connection aborted",
                "temporarily unavailable",
                "service unavailable",
            )
        ):
            return "transient"
        return "fatal"

    @staticmethod
    def _content_text(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content)

    @staticmethod
    def _prepare_messages(messages):
        prepared = []

        for message in messages:
            if isinstance(message, ChatMessage):
                role = message.role.value
                content = OpenRouterModel._content_text(message.content)
                tool_calls = getattr(message, "tool_calls", None)
                additional = getattr(message, "additional_kwargs", {}) or {}

                if role == "assistant" and tool_calls:
                    encoded_calls = []
                    for call in tool_calls:
                        function = call.function
                        arguments = function.arguments
                        if not isinstance(arguments, str):
                            arguments = json.dumps(
                                arguments,
                                ensure_ascii=False,
                            )
                        encoded_calls.append(
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": function.name,
                                    "arguments": arguments,
                                },
                            }
                        )
                    prepared.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": encoded_calls,
                        }
                    )
                    continue

                if role in {"tool", "function"}:
                    tool_call_id = additional.get("tool_call_id") or getattr(
                        message,
                        "tool_call_id",
                        None,
                    )
                    tool_name = additional.get("name") or getattr(
                        message,
                        "name",
                        None,
                    )
                    item = {
                        "role": "tool",
                        "content": content[-6000:],
                    }
                    if tool_call_id:
                        item["tool_call_id"] = tool_call_id
                    if tool_name:
                        item["name"] = tool_name
                    if tool_call_id:
                        prepared.append(item)
                    else:
                        prepared.append(
                            {
                                "role": "user",
                                "content": f"<tool_response>\n{content[-6000:]}\n</tool_response>",
                            }
                        )
                    continue

                if role in {"tool-response", "tool_call"}:
                    prepared.append(
                        {
                            "role": "user",
                            "content": f"<tool_response>\n{content[-6000:]}\n</tool_response>",
                        }
                    )
                    continue

                prepared.append(
                    {
                        "role": role if role in {"system", "user", "assistant"} else "user",
                        "content": content[-6000:],
                    }
                )
                continue

            if isinstance(message, dict):
                item = dict(message)
                role = item.get("role", "user")
                if role == "tool-response":
                    item["role"] = "user"
                    item["content"] = (
                        "<tool_response>\n"
                        + str(item.get("content", ""))[-6000:]
                        + "\n</tool_response>"
                    )
                elif role == "tool":
                    item["content"] = str(item.get("content", ""))[-6000:]
                elif role == "assistant" and item.get("tool_calls"):
                    pass
                else:
                    item["content"] = OpenRouterModel._content_text(
                        item.get("content", "")
                    )[-6000:]
                prepared.append(item)
                continue

            prepared.append(
                {
                    "role": "user",
                    "content": str(message)[-6000:],
                }
            )

        return prepared

    @staticmethod
    def _allowed_tool_names(stage):
        if stage == "search_company_evidence":
            return {"search_company_evidence", "final_answer"}
        if stage == "final_answer":
            return {"final_answer"}
        return {stage}

    def _tool_schemas(self, tools_to_call_from, stage):
        allowed = self._allowed_tool_names(stage)
        return [
            get_tool_json_schema(item)
            for item in (tools_to_call_from or [])
            if item.name in allowed
        ]

    def _chat_request(
        self,
        model_id,
        messages,
        tools,
        stop_sequences=None,
        response_format=None,
        max_tokens=256,
    ):
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if stop_sequences:
            payload["stop"] = stop_sequences

        if response_format:
            payload["response_format"] = response_format

        return self.client.chat.completions.create(**payload)

    def _to_chat_message(
        self,
        response,
        allowed_names,
        stage,
        fallback_text,
    ):
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        parsed_calls = []

        for call in tool_calls:
            function = call.function
            name = getattr(function, "name", None)
            if name not in allowed_names:
                continue

            arguments = getattr(function, "arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            if not isinstance(arguments, dict):
                arguments = {}

            parsed_calls.append(
                ChatMessageToolCall(
                    id=getattr(call, "id", f"call_{len(parsed_calls)}"),
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name=name,
                        arguments=arguments,
                    ),
                )
            )

        content = self._content_text(getattr(message, "content", None))

        if parsed_calls:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=parsed_calls,
            )

        if stage == "final_answer":
            name = "final_answer"
            arguments = {
                "answer": content.strip() or fallback_text or "Investigation completed."
            }
        elif stage == "search_company_evidence":
            name = "search_company_evidence"
            arguments = {
                "query": _xray_state().get("default_evidence_query", "")
            }
        elif stage == "company_search":
            name = "company_search"
            arguments = {
                "query": _xray_state().get("investigation_question", "")
            }
        else:
            name = stage
            arguments = {}

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_stage_fallback",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name=name,
                        arguments=arguments,
                    ),
                )
            ],
        )

    def _call_model(
        self,
        model_id,
        messages,
        tools,
        stop_sequences,
        response_format,
        max_tokens,
    ):
        last_error = None

        for attempt in range(2):
            try:
                return self._chat_request(
                    model_id=model_id,
                    messages=messages,
                    tools=tools,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                error_class = self._error_class(exc)

                if attempt == 0 and error_class in {
                    "rate_limit",
                    "server",
                    "transient",
                }:
                    time.sleep(self._retry_after(exc))
                    continue

                break

        raise last_error

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        if not _get_config_value("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Configure it in Streamlit Secrets or .env."
            )

        stage = _xray_state()["current_stage"]
        prepared = self._prepare_messages(messages)
        tools = self._tool_schemas(tools_to_call_from, stage)
        max_tokens = int(
            kwargs.get(
                "max_tokens",
                kwargs.get("max_new_tokens", 256),
            )
        )

        try:
            response = self._call_model(
                self.primary_model,
                prepared,
                tools,
                stop_sequences,
                response_format,
                max_tokens,
            )
            return self._to_chat_message(
                response,
                self._allowed_tool_names(stage),
                stage,
                "",
            )

        except Exception as primary_error:
            error_class = self._error_class(primary_error)

            if error_class in {"auth", "bad_request", "fatal"}:
                raise RuntimeError(
                    f"OpenRouter primary model failed permanently: {primary_error}"
                ) from primary_error

            try:
                response = self._call_model(
                    self.fallback_model,
                    prepared,
                    tools,
                    stop_sequences,
                    response_format,
                    max_tokens,
                )
                return self._to_chat_message(
                    response,
                    self._allowed_tool_names(stage),
                    stage,
                    "",
                )
            except Exception as fallback_error:
                if self._error_class(fallback_error) == "credit":
                    raise RuntimeError(
                        "OpenRouter credits are exhausted or the API key has "
                        "insufficient balance for both configured models."
                    ) from fallback_error

                raise RuntimeError(
                    "OpenRouter primary and fallback models failed. "
                    f"Primary: {primary_error}; Fallback: {fallback_error}"
                ) from fallback_error


def get_corporate_xray_agent():
    """Build the single session-scoped Corporate X-Ray agent."""
    state = _session()
    agent = state.get("_agent")

    if agent is not None:
        return agent

    if DEPLOYMENT_MODE == "cloud":
        api_key = _get_config_value("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when CORPORATE_XRAY_DEPLOYMENT=cloud."
            )
        model = OpenRouterModel(
            api_key=api_key,
            base_url=_get_config_value(
                "LLM_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            primary_model=_get_config_value(
                "LLM_MODEL_NAME",
                "qwen/qwen-2.5-72b-instruct",
            ),
            fallback_model=_get_config_value(
                "LLM_FALLBACK_MODEL_NAME",
                "qwen/qwen-2.5-7b-instruct",
            ),
        )
    elif DEPLOYMENT_MODE == "local":
        model_weights, tokenizer = load_qwen()
        model = LocalQwenModel(
            model=model_weights,
            tokenizer=tokenizer,
            max_new_tokens=96,
        )
    else:
        raise RuntimeError(
            "CORPORATE_XRAY_DEPLOYMENT must be 'local' or 'cloud'."
        )

    agent = ToolCallingAgent(
        tools=[
            company_search,
            company_profile,
            company_officers,
            company_pscs,
            company_filings,
            company_charges,
            company_insolvency,
            search_company_evidence,
        ],
        model=model,
        max_steps=12,
        verbosity_level=1,
        final_answer_checks=[investigation_complete],
    )

    state["_agent"] = agent
    return agent

# 35. MANAGEMENT ANALYSIS
# ============================================================

def analyze_management():

    officers = (
        _xray_data().get(
            "officers"
        )
        or []
    )

    appointments = [

        {

            "name":
                item.get(
                    "name"
                ),

            "role":
                item.get(
                    "role"
                ),

            "appointed_on":
                item.get(
                    "appointed_on"
                ),
        }

        for item
        in officers

        if item.get(
            "appointed_on"
        )
    ]

    resignations = [

        {

            "name":
                item.get(
                    "name"
                ),

            "role":
                item.get(
                    "role"
                ),

            "resigned_on":
                item.get(
                    "resigned_on"
                ),
        }

        for item
        in officers

        if item.get(
            "resigned_on"
        )
    ]

    appointments.sort(

        key=lambda item:
            item.get(
                "appointed_on"
            )
            or "",

        reverse=True,
    )

    resignations.sort(

        key=lambda item:
            item.get(
                "resigned_on"
            )
            or "",

        reverse=True,
    )

    return {

        "total_officers":
            len(
                officers
            ),

        "recent_appointments":
            appointments[:10],

        "recent_resignations":
            resignations[:10],
    }


# ============================================================
# 36. FILING ANALYSIS
# ============================================================

def analyze_filings():

    filings = (
        _xray_data().get(
            "filings"
        )
        or []
    )

    type_counts = {}

    for item in filings:

        filing_type = (
            item.get(
                "type"
            )
            or "UNKNOWN"
        )

        type_counts[
            filing_type
        ] = (
            type_counts.get(
                filing_type,
                0,
            )
            + 1
        )

    return {

        "total_filings":
            len(filings),

        "type_counts":
            type_counts,

        "recent_filings":
            filings[:10],
    }


# ============================================================
# 37. CHARGES ANALYSIS
# ============================================================

def analyze_charges():

    charges = (
        _xray_data().get(
            "charges"
        )
        or []
    )

    status_counts = {}

    for item in charges:

        status = (
            item.get(
                "status"
            )
            or "unknown"
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    return {

        "total_charges":
            len(charges),

        "status_counts":
            status_counts,

        "recent_charges":
            charges[:10],
    }


# ============================================================
# 38. FINAL REPORT
# ============================================================

def build_corporate_xray_report():

    profile = (
        _xray_data().get(
            "company_profile"
        )
        or {}
    )

    pscs = (
        _xray_data().get(
            "pscs"
        )
        or []
    )

    insolvency = (
        _xray_data().get(
            "insolvency"
        )
        or {}
    )

    return {

        "company_overview": {

            "name":
                profile.get(
                    "company_name"
                ),

            "number":
                profile.get(
                    "company_number"
                ),

            "status":
                profile.get(
                    "company_status"
                ),

            "type":
                profile.get(
                    "company_type"
                ),

            "created":
                profile.get(
                    "date_of_creation"
                ),

            "jurisdiction":
                profile.get(
                    "jurisdiction"
                ),

            "sic_codes":
                profile.get(
                    "sic_codes",
                    [],
                ),
        },

        "management":
            analyze_management(),

        "ownership": {

            "psc_count":
                len(pscs),

            "psc_records":
                pscs[:10],
        },

        "filing_activity":
            analyze_filings(),

        "charges":
            analyze_charges(),

        "insolvency": {

            "information_available":
                insolvency.get(
                    "available",
                    False,
                ),

            "case_count":
                len(
                    insolvency.get(
                        "cases",
                        [],
                    )
                ),
        },

        "documentary_evidence":
            _evidence()[:5],
    }


# ============================================================
# 39. MAIN APPLICATION ENTRY
# ============================================================

def run_corporate_xray(
    company_name,
):
    """
    Run complete one-agent Corporate X-Ray investigation.
    """

    company_name = (
        str(company_name)
        .strip()
    )

    if not company_name:

        raise ValueError(
            "Company name cannot be empty."
        )

    reset_corporate_xray_state()
    _xray_state()["investigation_question"] = company_name
    _xray_state()["default_evidence_query"] = (
        f"Recent Companies House documentary evidence for {company_name}: "
        "director appointments or changes, ownership or PSC changes, "
        "filing activity, registered charges, and insolvency-related filings."
    )
    _xray_state()["started_at"] = __import__("time").time()

    agent = get_corporate_xray_agent()

    prompt = f"""
Perform a complete Corporate X-Ray investigation of:

{company_name}

Follow the investigation workflow in order:

1. company_search
2. company_profile
3. company_officers
4. company_pscs
5. company_filings
6. company_charges
7. company_insolvency
8. search_company_evidence
9. final_answer

Rules:

- Identify the exact requested UK legal company.
- Use the exact company number returned by company_search.
- Never substitute a different company.
- Complete every structured investigation stage.
- Use official Companies House observations.
- After structured collection, use search_company_evidence.
- Evaluate whether the retrieved documentary evidence actually supports the investigative finding.
- If the evidence is insufficient, reformulate the query and search again.
- You may perform at most 3 evidence searches.
- Do not invent facts.
- Clearly distinguish missing information from confirmed information.
- Keep the final report factual and evidence-grounded.
"""

    result = agent.run(
        prompt
    )

    return {

        "agent_result":
            str(result),

        "report":
            build_corporate_xray_report(),

        "state":
            _xray_state().copy(),

        "evidence":
            list(
                _evidence()
            ),
    }


# ============================================================

# ============================================================
# 40. SYSTEM HEALTH CHECK
# ============================================================

def system_health_check():
    """Return system configuration without triggering model inference."""
    healthy, message = check_backend_health()

    return {
        "agents": 1,
        "tools": len(TOOL_NAMES),
        "tool_names": list(TOOL_NAMES),
        "model": (
            _get_config_value(
                "LLM_MODEL_NAME",
                "qwen/qwen-2.5-72b-instruct",
            )
            if DEPLOYMENT_MODE == "cloud"
            else MODEL_ID
        ),
        "fallback_model": (
            _get_config_value(
                "LLM_FALLBACK_MODEL_NAME",
                "qwen/qwen-2.5-7b-instruct",
            )
            if DEPLOYMENT_MODE == "cloud"
            else None
        ),
        "embedding_model": EMBEDDING_MODEL_ID,
        "reranker": RERANKER_MODEL_ID,
        "deployment_mode": DEPLOYMENT_MODE,
        "provider": "OpenRouter" if DEPLOYMENT_MODE == "cloud" else "local",
        "cloud_base_url": OPENROUTER_BASE_URL if DEPLOYMENT_MODE == "cloud" else None,
        "healthy": healthy,
        "health_message": message,
        "cuda_available": (
            __import__("torch").cuda.is_available()
            if DEPLOYMENT_MODE == "local"
            else False
        ),
        "gpu": (
            __import__("torch").cuda.get_device_name(0)
            if DEPLOYMENT_MODE == "local"
            and __import__("torch").cuda.is_available()
            else "CPU"
        ),
    }

# END OF CORPORATE X-RAY
