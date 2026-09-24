import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pymupdf
import requests
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from smolagents import ChatMessage, MessageRole, Model, ToolCallingAgent, tool
from smolagents.models import get_tool_json_schema


# ============================================================
# 1. CONFIGURATION
# ============================================================

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
MAX_EVIDENCE_ATTEMPTS = 3


def load_dotenv_file():
    """Load simple KEY=VALUE pairs from the local .env file."""
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

COMPANIES_HOUSE_API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()


def require_companies_house_api_key():
    """Return the configured Companies House API key or fail clearly."""
    key = os.getenv("COMPANIES_HOUSE_API_KEY", "").strip() or COMPANIES_HOUSE_API_KEY
    if not key:
        raise RuntimeError(
            "COMPANIES_HOUSE_API_KEY is missing. "
            "Create a local .env file or configure the environment variable "
            "before running Corporate X-Ray."
        )
    return key


# ============================================================
# 3. CORPORATE X-RAY WORKFLOW
# ============================================================

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


# ============================================================
# 4. INVESTIGATION STATE
# ============================================================

corporate_xray_state = {
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
    "started_at": None,
}


corporate_xray_data = {

    "company_search":
        None,

    "company_profile":
        None,

    "officers":
        None,

    "pscs":
        None,

    "filings":
        None,

    "charges":
        None,

    "insolvency":
        None,
}


corporate_xray_evidence = []


# ============================================================
# 5. RAG STATE
# ============================================================

rag_state = {

    "company_number":
        None,

    "chunks":
        [],

    "bm25":
        None,

    "embedding_model":
        None,

    "document_embeddings":
        None,

    "reranker":
        None,
}


# ============================================================
# 6. MODEL CACHE
# ============================================================

_qwen_model = None
_qwen_tokenizer = None
_agent = None


# ============================================================
# 7. RESET STATE
# ============================================================

def reset_corporate_xray_state():
    """Reset investigation, evidence and telemetry state for a new run."""
    corporate_xray_state.update({
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
        "started_at": None,
    })

    for key in corporate_xray_data:
        corporate_xray_data[key] = None

    corporate_xray_evidence.clear()

    rag_state["company_number"] = None
    rag_state["chunks"] = []
    rag_state["bm25"] = None
    rag_state["document_embeddings"] = None


# ============================================================
# 8. WORKFLOW HELPERS
# ============================================================

def advance_stage(completed_stage):
    """Advance the guarded structured workflow. Evidence remains agent-controlled."""
    if completed_stage == "search_company_evidence":
        if corporate_xray_state["evidence_attempts"] >= MAX_EVIDENCE_ATTEMPTS:
            corporate_xray_state["evidence_completed"] = True
            corporate_xray_state["current_stage"] = "final_answer"
        else:
            corporate_xray_state["current_stage"] = "search_company_evidence"
        return

    index = WORKFLOW.index(completed_stage)
    if index + 1 < len(WORKFLOW):
        corporate_xray_state["current_stage"] = WORKFLOW[index + 1]
    else:
        corporate_xray_state["current_stage"] = "final_answer"


def current_company_number():
    """
    Return the selected company number.
    """

    number = (
        corporate_xray_state.get(
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
        corporate_xray_data.get(
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

            0
            if item.get(
                "type"
            )
            in priority_types
            else 1,

            item.get(
                "date"
            )
            or "",
        )
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

        rag_state.update({

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

    if (
        rag_state[
            "embedding_model"
        ]
        is None
    ):

        rag_state[
            "embedding_model"
        ] = SentenceTransformer(

            EMBEDDING_MODEL_ID,

            device="cpu",
        )

    embeddings = (
        rag_state[
            "embedding_model"
        ].encode(

            [
                item["text"]
                for item
                in chunks
            ],

            normalize_embeddings=True,

            show_progress_bar=False,

            batch_size=32,
        )
    )

    if (
        rag_state[
            "reranker"
        ]
        is None
    ):

        rag_state[
            "reranker"
        ] = CrossEncoder(

            RERANKER_MODEL_ID,

            device="cpu",
        )

    rag_state.update({

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
        rag_state[
            "company_number"
        ]
        == company_number

        and rag_state[
            "chunks"
        ]
    ):

        return len(
            rag_state[
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

    chunks = rag_state[
        "chunks"
    ]

    if not chunks:
        return []

    query_tokens = tokenize_text(
        query
    )

    bm25_scores = (
        rag_state[
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

    query_embedding = (
        rag_state[
            "embedding_model"
        ].encode(
            query,
            normalize_embeddings=True,
        )
    )

    semantic_scores = np.dot(

        rag_state[
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
            rag_state[
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

    scores = (
        rag_state[
            "reranker"
        ].predict(
            pairs
        )
    )

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
    result = search_company(
        query,
        items_per_page=10,
    )

    if not result:
        return {
            "status": "not_found",
            "message": f"No Companies House result found for '{query}'.",
        }

    query_normalized = query.strip().upper()

    exact_matches = []
    other_matches = []

    for item in result:
        name = (
            item.get("company_name") or ""
        ).strip().upper()

        if name == query_normalized:
            exact_matches.append(item)
        else:
            other_matches.append(item)

    exact_matches.sort(
        key=lambda x: x.get("company_status") != "active"
    )

    other_matches.sort(
        key=lambda x: x.get("company_status") != "active"
    )

    matches = (
        exact_matches[:3]
        + other_matches[:2]
    )

    return {
        "status": "success",
        "query": query,
        "matches": matches,
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
        corporate_xray_state[
            "current_stage"
        ]
        != "company_profile"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                corporate_xray_state[
                    "current_stage"
                ],
        }

    company_number = (
        current_company_number()
    )

    result = get_company_profile(
        company_number
    )

    corporate_xray_data[
        "company_profile"
    ] = result

    corporate_xray_state[
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
        corporate_xray_state[
            "current_stage"
        ]
        != "company_officers"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                corporate_xray_state[
                    "current_stage"
                ],
        }

    result = get_officers(
        current_company_number()
    )

    corporate_xray_data[
        "officers"
    ] = result

    corporate_xray_state[
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
        corporate_xray_state[
            "current_stage"
        ]
        != "company_pscs"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                corporate_xray_state[
                    "current_stage"
                ],
        }

    result = get_pscs(
        current_company_number()
    )

    corporate_xray_data[
        "pscs"
    ] = result

    corporate_xray_state[
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
        corporate_xray_state[
            "current_stage"
        ]
        != "company_filings"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                corporate_xray_state[
                    "current_stage"
                ],
        }

    result = get_filing_history(
        current_company_number(),
        20,
    )

    corporate_xray_data[
        "filings"
    ] = result

    corporate_xray_state[
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
        corporate_xray_state[
            "current_stage"
        ]
        != "company_charges"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                corporate_xray_state[
                    "current_stage"
                ],
        }

    result = get_charges(
        current_company_number()
    )

    corporate_xray_data[
        "charges"
    ] = result

    corporate_xray_state[
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
        corporate_xray_state[
            "current_stage"
        ]
        != "company_insolvency"
    ):

        return {

            "status":
                "blocked",

            "required_stage":
                corporate_xray_state[
                    "current_stage"
                ],
        }

    result = get_insolvency(
        current_company_number()
    )

    corporate_xray_data[
        "insolvency"
    ] = result

    corporate_xray_state[
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
def search_company_evidence(query: str = "") -> list:
    """Search official filing documents using hybrid RAG and reranking.

    The tool is intentionally repeatable. The single agent can call it again
    with a refined query when the first evidence set is insufficient.
    """
    if corporate_xray_state["current_stage"] not in {
        "search_company_evidence",
        "final_answer",
    }:
        return [{
            "status": "blocked",
            "required_stage": corporate_xray_state["current_stage"],
        }]

    if corporate_xray_state["current_stage"] == "final_answer":
        return [{"status": "blocked", "message": "Evidence search is closed."}]

    attempts = corporate_xray_state["evidence_attempts"]
    if attempts >= MAX_EVIDENCE_ATTEMPTS:
        corporate_xray_state["evidence_completed"] = True
        corporate_xray_state["current_stage"] = "final_answer"
        return [{
            "status": "limit_reached",
            "message": "Maximum evidence-search attempts reached.",
        }]

    query = (query or "").strip()
    if not query:
        query = (
            corporate_xray_state.get("default_evidence_query")
            or "official filing evidence recent company activity"
        )

    corporate_xray_state["evidence_attempts"] += 1
    corporate_xray_state["evidence_queries"].append(query)

    try:
        chunk_count = ensure_company_rag_index(current_company_number())
        evidence = hybrid_search(query) if chunk_count else []
    except Exception as exc:
        evidence = [{
            "status": "evidence_unavailable",
            "message": str(exc)[:300],
        }]

    corporate_xray_evidence.clear()
    if isinstance(evidence, list):
        corporate_xray_evidence.extend(evidence)

    # Keep the stage open so the agent can decide whether to refine the query.
    # Only the hard retry cap forces transition to final_answer.
    if corporate_xray_state["evidence_attempts"] >= MAX_EVIDENCE_ATTEMPTS:
        corporate_xray_state["evidence_completed"] = True
        corporate_xray_state["current_stage"] = "final_answer"
    else:
        corporate_xray_state["current_stage"] = "search_company_evidence"

    return corporate_xray_evidence[:3]


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
        if not corporate_xray_state.get(key, False)
    ]

    if missing:
        raise ValueError(
            "Investigation incomplete. Missing steps: " + ", ".join(missing)
        )

    if not corporate_xray_state.get("evidence_attempts", 0):
        raise ValueError(
            "FINAL ANSWER REJECTED: documentary evidence search has not been attempted."
        )

    if not corporate_xray_state.get("evidence_completed"):
        # The evidence loop can finish by user/agent decision before the hard cap.
        # A non-empty evidence search is sufficient to permit synthesis; unsupported
        # claims are handled by the report-generation prompt and evidence metadata.
        corporate_xray_state["evidence_completed"] = True

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
        stage = corporate_xray_state["current_stage"]

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
                            arguments["query"] = corporate_xray_state.get("investigation_question", "")
                        if name == "search_company_evidence" and not arguments.get("query"):
                            arguments["query"] = corporate_xray_state.get("default_evidence_query", "")
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
            if corporate_xray_state.get("evidence_attempts", 0) >= MAX_EVIDENCE_ATTEMPTS:
                return {
                    "name": "final_answer",
                    "arguments": {
                        "answer": generated_text.strip() or "Investigation completed.",
                    },
                }
            return {
                "name": "search_company_evidence",
                "arguments": {
                    "query": corporate_xray_state.get("default_evidence_query", "")
                },
            }

        fallback_args = {}
        if stage == "company_search":
            fallback_args["query"] = corporate_xray_state.get("investigation_question", "")

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
            corporate_xray_state[
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
# 33. LOAD LOCAL QWEN
# ============================================================

def load_qwen():
    """
    Load Qwen2.5-3B.

    CUDA is preferred.
    4-bit NF4 is used on NVIDIA GPU.
    CPU fallback is available for environments without CUDA.
    """

    global _qwen_model
    global _qwen_tokenizer

    if (
        _qwen_model is not None
        and _qwen_tokenizer is not None
    ):

        return (
            _qwen_model,
            _qwen_tokenizer,
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(

            MODEL_ID,

            token=(
                HF_TOKEN
                or None
            ),
        )
    )

    # --------------------------------------------------------
    # NVIDIA CUDA PATH
    # --------------------------------------------------------

    if torch.cuda.is_available():

        quant_config = (
            BitsAndBytesConfig(

                load_in_4bit=
                    True,

                bnb_4bit_quant_type=
                    "nf4",

                bnb_4bit_compute_dtype=
                    torch.float16,

                bnb_4bit_use_double_quant=
                    True,
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(

                MODEL_ID,

                token=(
                    HF_TOKEN
                    or None
                ),

                device_map=
                    "auto",

                torch_dtype=
                    torch.float16,

                quantization_config=
                    quant_config,

                attn_implementation=
                    "sdpa",
            )
        )

    # --------------------------------------------------------
    # CPU FALLBACK
    # --------------------------------------------------------

    else:

        model = (
            AutoModelForCausalLM
            .from_pretrained(

                MODEL_ID,

                token=(
                    HF_TOKEN
                    or None
                ),

                device_map=
                    "cpu",

                torch_dtype=
                    torch.float32,
            )
        )

    _qwen_model = model
    _qwen_tokenizer = tokenizer

    return (
        _qwen_model,
        _qwen_tokenizer,
    )


# ============================================================
# 34. CREATE THE ONE AGENT
# ============================================================

def get_corporate_xray_agent():
    """
    Build the single Corporate X-Ray agent.
    """

    global _agent

    if _agent is not None:

        return _agent

    model, tokenizer = (
        load_qwen()
    )

    local_model = (
        LocalQwenModel(

            model=
                model,

            tokenizer=
                tokenizer,

            max_new_tokens=
                96,
        )
    )

    xray_tools = [

        company_search,

        company_profile,

        company_officers,

        company_pscs,

        company_filings,

        company_charges,

        company_insolvency,

        search_company_evidence,
    ]

    # Hard guarantee: exactly 8 tools.
    assert len(
        xray_tools
    ) == 8

    _agent = ToolCallingAgent(

        tools=
            xray_tools,

        model=
            local_model,

        max_steps=
            12,

        verbosity_level=
            1,

        final_answer_checks=[
            investigation_complete
        ],
    )

    return _agent


# ============================================================
# 35. MANAGEMENT ANALYSIS
# ============================================================

def analyze_management():

    officers = (
        corporate_xray_data.get(
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
        corporate_xray_data.get(
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
        corporate_xray_data.get(
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
        corporate_xray_data.get(
            "company_profile"
        )
        or {}
    )

    pscs = (
        corporate_xray_data.get(
            "pscs"
        )
        or []
    )

    insolvency = (
        corporate_xray_data.get(
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
            corporate_xray_evidence[:5],
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
    corporate_xray_state["investigation_question"] = company_name
    corporate_xray_state["started_at"] = __import__("time").time()

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
            corporate_xray_state.copy(),

        "evidence":
            list(
                corporate_xray_evidence
            ),
    }


# ============================================================
# 40. SYSTEM HEALTH CHECK
# ============================================================

def system_health_check():
    """
    Return system configuration and runtime health.
    """

    return {

        "agents":
            1,

        "tools":
            len(
                TOOL_NAMES
            ),

        "tool_names":
            list(
                TOOL_NAMES
            ),

        "model":
            MODEL_ID,

        "embedding_model":
            EMBEDDING_MODEL_ID,

        "reranker":
            RERANKER_MODEL_ID,

        "cuda_available":
            torch.cuda.is_available(),

        "gpu":
            (
                torch.cuda.get_device_name(
                    0
                )
                if torch.cuda.is_available()
                else "CPU"
            ),
    }


# ============================================================
# END OF CORPORATE X-RAY
# ============================================================
