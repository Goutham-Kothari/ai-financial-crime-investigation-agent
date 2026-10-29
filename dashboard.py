
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Financial Crime Investigation Agent",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Financial Crime Investigation Agent")
st.caption("Transaction risk scoring and evidence-based investigation")

st.info(
    "This tool supports analyst review. A model score is not proof of fraud."
)

st.sidebar.header("Investigation")
transaction_id = st.sidebar.number_input(
    "PaySim transaction ID",
    min_value=1,
    value=100,
    step=1,
)

if st.sidebar.button("Investigate Transaction", type="primary"):
    try:
        response = requests.post(
            f"{API_URL}/investigate-dataset",
            json={"transaction_id": int(transaction_id)},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            st.session_state["investigation_result"] = result
        else:
            st.error(
                f"API returned {response.status_code}: {response.text}"
            )

    except requests.RequestException as error:
        st.error(
            "Could not connect to the API. Start FastAPI in another "
            "terminal and try again. Details: " + str(error)
        )

result = st.session_state.get("investigation_result")

if result:
    transaction = result.get("transaction", {})
    evidence = result.get("evidence", {})

    st.subheader(
        f"Investigation — Transaction {result['dataset_transaction_id']}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", f"{result['risk_score']:.2%}")
    col2.metric("Risk Level", result["risk_level"])
    col3.metric("Transaction Type", transaction.get("type", "Unknown"))

    st.subheader("Transaction Details")

    st.dataframe(
        {
            "Field": [
                "Step",
                "Amount",
                "Origin Account",
                "Destination Account",
            ],
            "Value": [
                transaction.get("step"),
                transaction.get("amount"),
                transaction.get("origin_account"),
                transaction.get("destination_account"),
            ],
        },
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Investigation Summary")
    st.write(result.get("investigation_summary", "No summary available."))

    st.subheader("Evidence")

    evidence_fields = [
        "origin_balance_before",
        "origin_balance_after",
        "origin_balance_decrease",
        "destination_balance_before",
        "destination_balance_after",
        "destination_balance_change",
        "origin_amount_difference",
        "amount_matches_origin_decrease",
    ]

    evidence_table = [
        {"Field": key, "Value": evidence.get(key)}
        for key in evidence_fields
        if key in evidence
    ]

    if evidence_table:
        st.dataframe(
            evidence_table,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Findings")

    findings = evidence.get("findings", [])

    if findings:
        for finding in findings:
            st.write(
                f"**{finding.get('severity', 'info').upper()} — "
                f"{finding.get('code', 'FINDING')}**"
            )
            st.write(finding.get("finding", ""))
    else:
        st.write("No structured findings were returned.")

    st.subheader("Recommended Actions")

    for action in result.get("recommended_actions", []):
        st.markdown(f"- {action}")

    st.warning(result.get("warning", "Review results before taking action."))

    with st.expander("Offline evaluation label"):
        st.write(
            "Ground-truth label for testing only; it is not evidence "
            "available to a real-time investigator."
        )
        st.json(result.get("offline_evaluation_only", {}))
else:
    st.write(
        "Enter a transaction ID in the sidebar and select "
        "**Investigate Transaction** to begin."
    )
