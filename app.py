import streamlit as st
from datetime import datetime
from llm import analyze_customer, analyze_customer_behaviour
from chat_simulator import chat_simulator

st.set_page_config(
    page_title="Churn Copilot",
    page_icon="🎯",
    layout="wide"
)

@st.cache_data
def load_customer_analysis(idx):
    return analyze_customer(idx)

if "churn_data" not in st.session_state:
    with st.spinner("Analyze customer profile with ML & Qwen3..."):
        result = load_customer_analysis(idx=10)
        churn_proba = result["churn_prob"]
        reasoning = result["reasoning"]
        opening_chat = chat_simulator(churn_proba=churn_proba, reasoning=reasoning, chat_history=[])
        initial_behaviour = analyze_customer_behaviour(opening_chat)
        
        st.session_state.churn_data = {
            "churn_prob": churn_proba,
            "reasoning": reasoning,
            "recommendation": result["recommendation"],
            "behaviour": initial_behaviour
        }
        st.session_state.messages = [
                {"role": "assistant", "content": opening_chat, "time": datetime.now().strftime("%H.%M")}
            ]
    
data = st.session_state.churn_data

churn_proba = data['churn_prob']
reasoning = data['reasoning']
recommendation = data['recommendation']
behaviour = data['behaviour']

with st.expander(f"Churn Risk: {churn_proba}% | Customer Behaviour: {behaviour}", expanded=True, key="info_panel"):
        col1, col2, col3 = st.columns([1, 2, 1], vertical_alignment="center")
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding-top: 10px;">
                <p style="margin: -10px 0 0 0; font-size: 20px; color: #ffffff;">Churn Risk Probability:</p>
                <h1 style="margin: -20px 0 0 25px; font-size: 60px; color: #ffffff; font-weight: 700;">{churn_proba}%</h1>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="border-left: 2px solid #a0a0a0; border-right: 2px solid #a0a0a0; padding: 0px 25px 15px 25px; margin-top: -10px;">
                <p style="margin: 0; font-size: 18px; font-weight: 600;">Reasoning:</p>
                <p style="margin: 2px 0 12px 0; font-size: 14px; color: #d0d3d8;">
                    {reasoning}
                </p>
                <p style="margin: 0; font-size: 18px; font-weight: 600;">Recommendation:</p>
                <p style="margin: 2px 0 0 0; font-size: 14px; color: #d0d3d8;">
                    {recommendation}
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding-top: 10px;">
                <p style="margin: -10px 0 0 0; font-size: 20px; color: #ffffff;">Customer Behaviour:</p>
                <h1 style="margin: -20px 0 0 25px; font-size: 46px; color: #ffffff; font-weight: 700;">{behaviour}</h1>
            </div>
            """, unsafe_allow_html=True)
    
panel_is_open = st.session_state.get("info_panel", True)
chat_max_height = 425 if panel_is_open else 650

chat_html = f'<div class="chat-scroll-box" style="max-height: {chat_max_height}px;">'

for msg in reversed(st.session_state.messages):
    if msg["role"] == "user":
        chat_html += f'<div class="chat-row right"><div class="bubble user-bubble"><span>{msg["content"]}</span><span class="timestamp">{msg["time"]}</span></div></div>'
    else:
        chat_html += f'<div class="chat-row left"><div class="bubble bot-bubble"><span>{msg["content"]}</span><span class="timestamp">{msg["time"]}</span></div></div>'

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

if prompt := st.chat_input("your text"):
    now = datetime.now().strftime("%H.%M")
    
    st.session_state.messages.append({"role": "user", "content": prompt, "time": now})
    
    with st.spinner("Customer is Typing..."):
        bot_reply = chat_simulator(
            churn_proba=st.session_state.churn_data["churn_prob"],
            reasoning=st.session_state.churn_data["reasoning"],
            chat_history=st.session_state.messages
        )
        new_behaviour = analyze_customer_behaviour(bot_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "time": datetime.now().strftime("%H.%M")
    })
    st.session_state.churn_data["behaviour"] = new_behaviour
    
    st.rerun()

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background: linear-gradient(180deg, #121212 0%, #3a3a3a 65%, #7a7a7a 100%);
    }
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #2b2b2b 0%, #525252 60%, #8c8c8c 100%) !important;
        border: none !important;
        border-radius: 0px 0px 30px 30px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stExpander"] summary {
        background-color: transparent !important;
    }
    div[data-testid="stExpander"] details {
       border: none !important;
       border-radius: 0px 0px 30px 30px !important;
    }
    div[data-testid="stExpander"] summary {
        border-bottom: none !important;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        opacity: 1 !important;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    div[data-testid="stBottom"], 
    div[data-testid="stBottom"] > div {
        background-color: transparent !important;
        background: none !important;
    }
    div[data-testid="stChatInput"] {
        max-width: 750px !important;
        margin: 0 auto !important;
    }
    div[data-testid="stChatInput"] > div {
        background-color: #dcdcdc !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 4px 15px !important;
    }
    div[data-testid="stChatInput"] textarea {
       color: #1e1e1e !important;
       background-color: transparent !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #666666 !important;
    }
    div[data-testid="stChatInput"] button,
    div[data-testid="stChatInput"] button:not(:disabled),
    div[data-testid="stChatInput"] button:hover {
        background-color: transparent !important;
        border: none !important;
        color: #1e1e1e !important; 
    }
    .chat-scroll-box {
        max-width: 750px;
        margin: 0px auto 10px auto !important;
        padding: 0 15px;
        display: flex;
        flex-direction: column-reverse;
        gap: 14px;
        max-height: 425px;
        overflow-y: auto;
    }
    .chat-row {
        display: flex;
        width: 100%;
        max-width: 750px;
        margin: 0 auto 0px auto;
        padding: 0;
        flex-shrink: 0;
    }
    .chat-row.left {
        justify-content: flex-start;
    }
    .chat-row.right {
        justify-content: flex-end;
    }
    .bubble {
        padding: 12px 18px;
        border-radius: 20px;
        max-width: 65%;
        font-size: 16px;
        line-height: 1.4;
        display: flex;
        align-items: flex-end;
        gap: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }
    .bot-bubble {
        background-color: #2f3136;
        color: #ffffff;
        border-bottom-left-radius: 4px;
    }
    .user-bubble {
        background-color: #e4e6eb;
        color: #1c1e21;
        border-bottom-right-radius: 4px;
    }
    .timestamp {
        font-size: 11px;
        opacity: 0.65;
        white-space: nowrap;
        margin-bottom: -2px;
    }
    </style>
    """, unsafe_allow_html=True)