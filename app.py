import html

import streamlit as st

import corporate_xray_agentic_rag as backend


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Corporate X-Ray",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "investigation_result" not in st.session_state:
    st.session_state.investigation_result = None

if "investigation_error" not in st.session_state:
    st.session_state.investigation_error = None

if "company_name" not in st.session_state:
    st.session_state.company_name = "REVOLUT LTD"


# ============================================================
# HELPERS
# ============================================================

def esc(value):
    return html.escape(str(value)) if value is not None else ""


def safe_list(value):
    return value if isinstance(value, list) else []


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
    <style>

    /* ================================
       APP
       ================================ */

    .stApp {
        background:
            radial-gradient(
                circle at 88% 0%,
                rgba(79,70,229,0.08),
                transparent 30%
            ),
            linear-gradient(
                180deg,
                #f8fafc 0%,
                #f1f5f9 100%
            );
    }

    .main .block-container {
        max-width: 1450px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: rgba(248,250,252,0.96) !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }


    /* ================================
       SIDEBAR
       ================================ */

    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {
        color: #cbd5e1 !important;
    }


    /* ================================
       NATIVE INPUTS
       ================================ */

    div[data-testid="stTextInput"] label {
        color: #334155 !important;
        font-size: 0.76rem !important;
        font-weight: 800 !important;
    }

    div[data-testid="stTextInput"] input {
        min-height: 50px !important;
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-size: 0.84rem !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.10) !important;
    }

    div.stButton > button {
        min-height: 50px !important;
        width: 100% !important;
        border: none !important;
        border-radius: 12px !important;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #6366f1
        ) !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        box-shadow: 0 8px 18px rgba(79,70,229,0.22) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            135deg,
            #4338ca,
            #4f46e5
        ) !important;
    }


    /* ================================
       TABS
       ================================ */

    button[data-baseweb="tab"] {
        color: #64748b !important;
        font-weight: 800 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4f46e5 !important;
    }


    /* ================================
       STATUS
       ================================ */

    div[data-testid="stStatusWidget"] {
        border-radius: 14px !important;
    }


    /* ================================
       MOBILE
       ================================ */

    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    </style>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### Corporate X-Ray")
    st.caption("Autonomous UK company intelligence")

    st.divider()

    st.markdown("**INVESTIGATION PIPELINE**")

    for item in [
        "○ Company identity",
        "○ Company profile",
        "○ Officers",
        "○ PSC / ownership",
        "○ Filing history",
        "○ Charges",
        "○ Insolvency",
        "○ Documentary evidence",
    ]:
        st.markdown(item)

    st.divider()

    st.markdown("**SYSTEM**")

    try:
        health = backend.system_health_check()

        st.markdown("✓ 1 AI agent")
        st.markdown("✓ 8 investigation tools")

        if health.get("cuda_available"):
            st.markdown("✓ CUDA enabled")
        else:
            st.markdown("○ CPU mode")

    except Exception:
        st.markdown("○ Backend status unavailable")

    st.divider()

    st.caption(
        "Source: Companies House public data."
    )

    st.caption(
        "Corporate X-Ray is an intelligence-support "
        "system for company research and evidence "
        "discovery. It is not legal or accounting advice."
    )


# ============================================================
# TOP NAV
# ============================================================

st.html(
    """
    <div style="
        width:100%;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:20px;
        padding:14px 18px;
        margin-bottom:30px;
        border:1px solid #e2e8f0;
        border-radius:18px;
        background:#ffffff;
        box-shadow:0 8px 26px rgba(15,23,42,0.05);
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:12px;
        ">

            <div style="
                width:40px;
                height:40px;
                display:flex;
                align-items:center;
                justify-content:center;
                border-radius:12px;
                background:linear-gradient(
                    135deg,
                    #4338ca,
                    #6366f1
                );
                color:white;
                font-weight:900;
                font-size:16px;
            ">
                ◈
            </div>

            <div>

                <div style="
                    color:#0f172a;
                    font-weight:900;
                    font-size:16px;
                ">
                    Corporate X-Ray
                </div>

                <div style="
                    color:#64748b;
                    font-size:11px;
                    margin-top:2px;
                ">
                    UK corporate intelligence
                </div>

            </div>

        </div>


        <div style="
            display:flex;
            gap:8px;
            flex-wrap:wrap;
            justify-content:flex-end;
        ">

            <div style="
                padding:7px 12px;
                border-radius:999px;
                background:#eef2ff;
                border:1px solid #c7d2fe;
                color:#4338ca;
                font-size:10px;
                font-weight:900;
            ">
                1 AI AGENT
            </div>

            <div style="
                padding:7px 12px;
                border-radius:999px;
                background:#ffffff;
                border:1px solid #e2e8f0;
                color:#475569;
                font-size:10px;
                font-weight:900;
            ">
                8 TOOLS
            </div>

            <div style="
                padding:7px 12px;
                border-radius:999px;
                background:#ffffff;
                border:1px solid #e2e8f0;
                color:#475569;
                font-size:10px;
                font-weight:900;
            ">
                COMPANIES HOUSE
            </div>

        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div style="
        padding:0 0 6px 0;
    ">

        <div style="
            color:#4f46e5;
            font-size:11px;
            font-weight:900;
            letter-spacing:2.5px;
            text-transform:uppercase;
            margin-bottom:10px;
        ">
            CORPORATE X-RAY
        </div>

        <div style="
            color:#0b1220;
            font-size:clamp(40px,5vw,68px);
            line-height:0.98;
            letter-spacing:-3px;
            font-weight:950;
            margin:0;
        ">
            UK Company
            <span style="color:#4f46e5;">
                Due-Diligence
            </span>
            Engine
        </div>

        <div style="
            max-width:980px;
            margin-top:18px;
            color:#475569;
            font-size:15px;
            line-height:1.75;
        ">
            Investigate UK companies using official Companies House
            records, documentary evidence retrieval, hybrid RAG,
            reranking and one autonomous AI agent.
        </div>

    </div>
    """
)


# ============================================================
# INVESTIGATION CARD
# ============================================================

st.html(
    """
    <div style="
        margin-top:26px;
        padding:22px;
        border:1px solid #dfe6ef;
        border-radius:18px;
        background:#ffffff;
        box-shadow:0 12px 32px rgba(15,23,42,0.05);
    ">

        <div style="
            color:#0f172a;
            font-size:17px;
            font-weight:900;
        ">
            Start an investigation
        </div>

        <div style="
            margin-top:5px;
            color:#64748b;
            font-size:12px;
            line-height:1.6;
        ">
            Enter the exact UK legal company name.
            Corporate X-Ray identifies the company and
            executes the investigation pipeline automatically.
        </div>

    </div>
    """
)


# ============================================================
# INPUT
# ============================================================

input_col, button_col = st.columns(
    [5.2, 1],
    gap="medium",
)

with input_col:
    company_name = st.text_input(
        "Company name",
        value=st.session_state.company_name,
        placeholder="e.g. REVOLUT LTD",
    ).strip()

with button_col:

    st.write("")

    run_clicked = st.button(
        "RUN X-RAY",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# RUN INVESTIGATION
# ============================================================

if run_clicked:

    if not company_name:

        st.session_state.investigation_result = None
        st.session_state.investigation_error = (
            "Please enter a UK company name."
        )

    else:

        st.session_state.company_name = company_name
        st.session_state.investigation_result = None
        st.session_state.investigation_error = None

        with st.status(
            f"Running Corporate X-Ray for {company_name}...",
            expanded=True,
        ) as status:

            try:

                st.write(
                    "Connecting to official Companies House data..."
                )

                result = backend.run_corporate_xray(
                    company_name
                )

                st.session_state.investigation_result = result

                status.update(
                    label="✓ Investigation completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:

                st.session_state.investigation_error = str(
                    exc
                )

                status.update(
                    label="Investigation failed",
                    state="error",
                    expanded=True,
                )


# ============================================================
# ERROR
# ============================================================

if st.session_state.investigation_error:

    st.error(
        st.session_state.investigation_error
    )


# ============================================================
# RESULT
# ============================================================

result = st.session_state.investigation_result

if result:

    report = result.get("report", {}) or {}
    state = result.get("state", {}) or {}
    evidence = result.get("evidence", []) or []

    company = report.get(
        "company_overview",
        {},
    ) or {}

    management = report.get(
        "management",
        {},
    ) or {}

    ownership = report.get(
        "ownership",
        {},
    ) or {}

    filing_activity = report.get(
        "filing_activity",
        {},
    ) or {}

    charges = report.get(
        "charges",
        {},
    ) or {}

    insolvency = report.get(
        "insolvency",
        {},
    ) or {}


    # ========================================================
    # RESULT HEADER
    # ========================================================

    company_display = esc(
        company.get("name") or "Company"
    )

    number_display = esc(
        company.get("number") or "N/A"
    )

    st.html(
        f"""
        <div style="
            margin-top:28px;
            margin-bottom:14px;
            padding:18px 20px;
            border:1px solid #dfe6ef;
            border-radius:16px;
            background:#ffffff;
            box-shadow:0 8px 24px rgba(15,23,42,0.045);
        ">

            <div style="
                color:#4f46e5;
                font-size:10px;
                font-weight:900;
                letter-spacing:1.7px;
            ">
                INVESTIGATION RESULT
            </div>

            <div style="
                margin-top:5px;
                color:#0f172a;
                font-size:21px;
                font-weight:950;
            ">
                {company_display}
            </div>

            <div style="
                margin-top:3px;
                color:#64748b;
                font-size:11px;
            ">
                Companies House number: {number_display}
            </div>

        </div>
        """
    )


    # ========================================================
    # METRICS
    # ========================================================

    metric_cols = st.columns(5)

    metric_values = [
        (
            "STATUS",
            company.get("status") or "N/A",
        ),
        (
            "OFFICERS",
            management.get(
                "total_officers"
            ) or 0,
        ),
        (
            "PSC",
            ownership.get(
                "psc_count"
            ) or 0,
        ),
        (
            "FILINGS",
            filing_activity.get(
                "total_filings"
            ) or 0,
        ),
        (
            "CHARGES",
            charges.get(
                "total_charges"
            ) or 0,
        ),
    ]

    for col, (label, value) in zip(
        metric_cols,
        metric_values,
    ):

        with col:

            st.html(
                f"""
                <div style="
                    min-height:96px;
                    padding:15px;
                    border:1px solid #e2e8f0;
                    border-radius:15px;
                    background:#ffffff;
                    box-shadow:0 7px 22px rgba(15,23,42,0.04);
                ">

                    <div style="
                        color:#64748b;
                        font-size:9px;
                        font-weight:900;
                        letter-spacing:1.1px;
                    ">
                        {esc(label)}
                    </div>

                    <div style="
                        margin-top:7px;
                        color:#0f172a;
                        font-size:19px;
                        font-weight:950;
                    ">
                        {esc(value)}
                    </div>

                </div>
                """
            )


    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(
        [
            "Overview",
            "Management",
            "Ownership",
            "Filings",
            "Charges",
            "Insolvency",
            "Evidence",
        ]
    )


    # ========================================================
    # OVERVIEW
    # ========================================================

    with tabs[0]:

        left, right = st.columns(
            2,
            gap="medium",
        )

        with left:

            st.html(
                """
                <div style="
                    padding:18px;
                    border:1px solid #e2e8f0;
                    border-radius:15px;
                    background:#ffffff;
                ">

                    <div style="
                        color:#0f172a;
                        font-size:15px;
                        font-weight:900;
                    ">
                        Company identity
                    </div>

                    <div style="
                        margin-top:4px;
                        color:#64748b;
                        font-size:11px;
                    ">
                        Core legal information.
                    </div>

                </div>
                """
            )

            rows = [
                (
                    "Legal name",
                    company.get("name") or "N/A",
                ),
                (
                    "Company number",
                    company.get("number") or "N/A",
                ),
                (
                    "Status",
                    company.get("status") or "N/A",
                ),
                (
                    "Type",
                    company.get("type") or "N/A",
                ),
                (
                    "Created",
                    company.get("created") or "N/A",
                ),
                (
                    "Jurisdiction",
                    company.get("jurisdiction") or "N/A",
                ),
                (
                    "SIC codes",
                    ", ".join(
                        str(x)
                        for x in (
                            company.get(
                                "sic_codes"
                            )
                            or []
                        )
                    ) or "N/A",
                ),
            ]

            for label, value in rows:

                st.html(
                    f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        gap:20px;
                        padding:10px 0;
                        border-bottom:1px solid #f1f5f9;
                    ">

                        <span style="
                            color:#64748b;
                            font-size:11px;
                        ">
                            {esc(label)}
                        </span>

                        <strong style="
                            color:#0f172a;
                            font-size:11px;
                            text-align:right;
                        ">
                            {esc(value)}
                        </strong>

                    </div>
                    """
                )


        with right:

            st.html(
                """
                <div style="
                    padding:18px;
                    border:1px solid #e2e8f0;
                    border-radius:15px;
                    background:#ffffff;
                ">

                    <div style="
                        color:#0f172a;
                        font-size:15px;
                        font-weight:900;
                    ">
                        Investigation pipeline
                    </div>

                    <div style="
                        margin-top:4px;
                        color:#64748b;
                        font-size:11px;
                    ">
                        Autonomous execution status.
                    </div>

                </div>
                """
            )

            stages = [
                (
                    "Company search",
                    "company_search_completed",
                ),
                (
                    "Company profile",
                    "company_profile_completed",
                ),
                (
                    "Officers",
                    "officers_completed",
                ),
                (
                    "PSC / ownership",
                    "pscs_completed",
                ),
                (
                    "Filing history",
                    "filings_completed",
                ),
                (
                    "Charges",
                    "charges_completed",
                ),
                (
                    "Insolvency",
                    "insolvency_completed",
                ),
                (
                    "Documentary evidence",
                    "evidence_completed",
                ),
            ]

            for label, key in stages:

                completed = bool(
                    state.get(
                        key,
                        False,
                    )
                )

                icon = "✓" if completed else "○"

                st.html(
                    f"""
                    <div style="
                        display:flex;
                        align-items:center;
                        gap:9px;
                        padding:8px 0;
                        border-bottom:1px solid #f1f5f9;
                    ">

                        <div style="
                            width:23px;
                            height:23px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            border-radius:7px;
                            background:#ecfdf5;
                            color:#047857;
                            font-size:11px;
                            font-weight:900;
                        ">
                            {icon}
                        </div>

                        <div style="
                            color:#334155;
                            font-size:11px;
                            font-weight:750;
                        ">
                            {esc(label)}
                        </div>

                    </div>
                    """
                )


    # ========================================================
    # MANAGEMENT
    # ========================================================

    with tabs[1]:

        st.markdown("### Management activity")

        appointments = safe_list(
            management.get(
                "recent_appointments",
                [],
            )
        )

        resignations = safe_list(
            management.get(
                "recent_resignations",
                [],
            )
        )

        if appointments:

            for item in appointments:

                st.html(
                    f"""
                    <div style="
                        margin-bottom:10px;
                        padding:15px;
                        border:1px solid #e2e8f0;
                        border-radius:14px;
                        background:#ffffff;
                    ">

                        <div style="
                            color:#0f172a;
                            font-size:13px;
                            font-weight:900;
                        ">
                            {esc(item.get("name") or "Unknown")}
                        </div>

                        <div style="
                            margin-top:4px;
                            color:#64748b;
                            font-size:11px;
                        ">
                            {esc(item.get("role") or "Role unavailable")}
                            · Appointed
                            {esc(item.get("appointed_on") or "N/A")}
                        </div>

                    </div>
                    """
                )

        else:

            st.info(
                "No appointment records available."
            )

        if resignations:

            st.markdown("### Recent resignations")

            for item in resignations:

                st.html(
                    f"""
                    <div style="
                        margin-bottom:10px;
                        padding:15px;
                        border:1px solid #e2e8f0;
                        border-radius:14px;
                        background:#ffffff;
                    ">

                        <div style="
                            color:#0f172a;
                            font-size:13px;
                            font-weight:900;
                        ">
                            {esc(item.get("name") or "Unknown")}
                        </div>

                        <div style="
                            margin-top:4px;
                            color:#64748b;
                            font-size:11px;
                        ">
                            {esc(item.get("role") or "Role unavailable")}
                            · Resigned
                            {esc(item.get("resigned_on") or "N/A")}
                        </div>

                    </div>
                    """
                )


    # ========================================================
    # OWNERSHIP
    # ========================================================

    with tabs[2]:

        st.markdown(
            "### People with Significant Control"
        )

        psc_records = safe_list(
            ownership.get(
                "psc_records",
                [],
            )
        )

        if psc_records:

            for psc in psc_records:

                controls = safe_list(
                    psc.get(
                        "nature_of_control",
                        [],
                    )
                )

                control_text = (
                    ", ".join(
                        str(x)
                        for x in controls
                    )
                    if controls
                    else "Not specified"
                )

                st.html(
                    f"""
                    <div style="
                        margin-bottom:10px;
                        padding:15px;
                        border:1px solid #e2e8f0;
                        border-radius:14px;
                        background:#ffffff;
                    ">

                        <div style="
                            color:#0f172a;
                            font-size:13px;
                            font-weight:900;
                        ">
                            {esc(psc.get("name") or "Unknown")}
                        </div>

                        <div style="
                            margin-top:4px;
                            color:#64748b;
                            font-size:11px;
                        ">
                            {esc(psc.get("kind") or "Type unavailable")}
                        </div>

                        <div style="
                            margin-top:8px;
                            color:#334155;
                            font-size:11px;
                            line-height:1.6;
                        ">
                            <strong>
                                Nature of control:
                            </strong>
                            {esc(control_text)}
                        </div>

                    </div>
                    """
                )

        else:

            st.info(
                "No PSC records were returned."
            )


    # ========================================================
    # FILINGS
    # ========================================================

    with tabs[3]:

        st.markdown(
            "### Recent filing activity"
        )

        recent_filings = safe_list(
            filing_activity.get(
                "recent_filings",
                [],
            )
        )

        if recent_filings:

            for filing in recent_filings:

                st.html(
                    f"""
                    <div style="
                        margin-bottom:10px;
                        padding:15px;
                        border:1px solid #e2e8f0;
                        border-radius:14px;
                        background:#ffffff;
                    ">

                        <div style="
                            display:flex;
                            justify-content:space-between;
                            gap:15px;
                        ">

                            <strong style="
                                color:#0f172a;
                                font-size:12px;
                            ">
                                {esc(filing.get("type") or "Filing")}
                            </strong>

                            <span style="
                                color:#64748b;
                                font-size:10px;
                            ">
                                {esc(filing.get("date") or "N/A")}
                            </span>

                        </div>

                        <div style="
                            margin-top:6px;
                            color:#475569;
                            font-size:11px;
                            line-height:1.5;
                        ">
                            {
                                esc(
                                    filing.get(
                                        "description"
                                    )
                                    or "No description available."
                                )
                            }
                        </div>

                        <div style="
                            margin-top:5px;
                            color:#94a3b8;
                            font-size:10px;
                        ">
                            Category:
                            {esc(filing.get("category") or "N/A")}
                        </div>

                    </div>
                    """
                )

        else:

            st.info(
                "No filing records were returned."
            )


    # ========================================================
    # CHARGES
    # ========================================================

    with tabs[4]:

        st.markdown("### Registered charges")

        status_counts = (
            charges.get(
                "status_counts",
                {},
            )
            or {}
        )

        if status_counts:

            charge_cols = st.columns(
                min(
                    4,
                    len(status_counts),
                )
            )

            for col, (
                label,
                value,
            ) in zip(
                charge_cols,
                status_counts.items(),
            ):

                with col:

                    st.html(
                        f"""
                        <div style="
                            padding:15px;
                            border:1px solid #e2e8f0;
                            border-radius:14px;
                            background:#ffffff;
                        ">

                            <div style="
                                color:#64748b;
                                font-size:9px;
                                font-weight:900;
                                text-transform:uppercase;
                                letter-spacing:1px;
                            ">
                                {esc(label)}
                            </div>

                            <div style="
                                margin-top:7px;
                                color:#0f172a;
                                font-size:20px;
                                font-weight:950;
                            ">
                                {esc(value)}
                            </div>

                        </div>
                        """
                    )

        else:

            st.info(
                "No charge records were returned."
            )


    # ========================================================
    # INSOLVENCY
    # ========================================================

    with tabs[5]:

        st.markdown("### Insolvency information")

        case_count = (
            insolvency.get(
                "case_count",
                0,
            )
            or 0
        )

        if case_count == 0:

            st.success(
                "No insolvency case records were returned "
                "for this investigation."
            )

        else:

            st.warning(
                f"{case_count} insolvency case record(s) returned."
            )


    # ========================================================
    # EVIDENCE
    # ========================================================

    with tabs[6]:

        st.markdown("### Documentary evidence")

        valid_evidence = [
            item
            for item in safe_list(evidence)
            if not item.get("status")
        ]

        if valid_evidence:

            for item in valid_evidence:

                st.html(
                    f"""
                    <div style="
                        padding:16px;
                        margin-bottom:10px;
                        border:1px solid #e2e8f0;
                        border-left:4px solid #6366f1;
                        border-radius:12px;
                        background:#ffffff;
                    ">

                        <div style="
                            color:#64748b;
                            font-size:10px;
                            font-weight:850;
                            margin-bottom:7px;
                        ">
                            Filing date:
                            {esc(item.get("filing_date") or "N/A")}
                            &nbsp; · &nbsp;
                            Page:
                            {esc(item.get("page") or "N/A")}
                            &nbsp; · &nbsp;
                            Category:
                            {esc(item.get("category") or "N/A")}
                        </div>

                        <div style="
                            color:#334155;
                            font-size:11px;
                            line-height:1.65;
                            white-space:pre-wrap;
                        ">
                            {
                                esc(
                                    item.get(
                                        "evidence",
                                        "No evidence text available.",
                                    )
                                )
                            }
                        </div>

                    </div>
                    """
                )

        else:

            st.info(
                "No documentary evidence was returned."
            )

        with st.expander(
            "Technical investigation output"
        ):

            st.write(
                result.get(
                    "agent_result",
                    "",
                )
            )


# ============================================================
# EMPTY STATE
# ============================================================

if (
    result is None
    and not st.session_state.investigation_error
):

    st.html(
        """
        <div style="
            margin-top:26px;
            padding:42px 24px;
            text-align:center;
            border:1px solid #e2e8f0;
            border-radius:16px;
            background:#ffffff;
            box-shadow:0 6px 18px rgba(15,23,42,0.035);
        ">

            <div style="
                width:48px;
                height:48px;
                margin:0 auto 12px;
                display:flex;
                align-items:center;
                justify-content:center;
                border-radius:14px;
                background:#eef2ff;
                color:#4f46e5;
                font-size:18px;
                font-weight:900;
            ">
                ◈
            </div>

            <div style="
                color:#0f172a;
                font-size:17px;
                font-weight:900;
            ">
                Ready for investigation
            </div>

            <div style="
                max-width:700px;
                margin:8px auto 0;
                color:#64748b;
                font-size:11px;
                line-height:1.7;
            ">
                Enter a UK company name above. Corporate X-Ray
                identifies the company, retrieves official records,
                analyses management, ownership, filings, charges
                and insolvency information, and retrieves
                documentary evidence.
            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        margin-top:38px;
        padding-top:14px;
        border-top:1px solid #e2e8f0;
        text-align:center;
        color:#94a3b8;
        font-size:10px;
    ">
        Corporate X-Ray · 1 AI Agent · 8 Tools ·
        Companies House · Hybrid RAG · Documentary Evidence
    </div>
    """
)