import streamlit as st
from transformers import pipeline
import openai
import logging

# --------- Page Configuration ---------
st.set_page_config(
    page_title="WarmTalk - Elderly Companion",
    page_icon="🧓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------- Warm Theme CSS ---------
st.markdown("""
    <style>
    :root {
        --primary: #FF9E80;
        --secondary: #FFCC80;
        --background: #FFF8F0;
        --text: #5D4037;
    }
    body {
        background-color: var(--background);
        color: var(--text);
    }
    .stTextArea textarea {
        font-size: 1.1rem;
        border: 2px solid var(--primary) !important;
        border-radius: 12px !important;
    }
    .stButton button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 1rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #FF7043 !important;
        transform: scale(1.02);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text) !important;
    }
    .stChatMessage {
        border-radius: 16px !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stSidebar"] {
        background-color: #FFF3E0 !important;
    }
    .stAlert {
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --------- Initialization Functions ---------
@st.cache_resource
def load_emotion_model():
    """Load and cache the emotion classification model"""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion"
    )

def initialize_session():
    """Initialize all required session state variables"""
    required_keys = {
        'conversation': [],
        'api_key': "",
        'emotion_model': load_emotion_model(),
        'user_input_widget': ""  # Changed from user_input to avoid conflict
    }
    
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --------- Content Moderation ---------
def contains_sensitive_content(text):
    """Check if text contains any sensitive words/phrases"""
    sensitive_words = {
        "violence": ["kill", "murder", "attack", "hurt"],
        "self-harm": ["suicide", "end my life", "harm myself"],
        "medical": ["die", "death", "terminal illness"]
    }
    
    text_lower = text.lower()
    for category, words in sensitive_words.items():
        if any(word in text_lower for word in words):
            return category
    return None

# --------- AI Response Generation ---------
def generate_ai_response():
    """Generate AI response using conversation history"""
    if not st.session_state.api_key:
        st.error("API key not set")
        return None

    try:
        client = openai.OpenAI(api_key=st.session_state.api_key)

        # Build conversation history
        messages = [{
            "role": "system",
            "content": """You are WarmTalk, a gentle companion for seniors.
            Respond with empathy and kindness in 2-3 short sentences.
            Use simple language and a warm tone."""
        }]

        for turn in st.session_state.conversation:
            role = "user" if turn['role'] == 'user' else "assistant"
            content = turn['content']
            if role == "user":
                emotion = st.session_state.emotion_model(content)[0]['label']
                content = f"[Feeling: {emotion}] {content}"
            messages.append({"role": role, "content": content})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        return response.choices[0].message.content

    except openai.AuthenticationError:
        st.error("Invalid API key - please check your settings")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# --------- Input Handling ---------
def handle_user_input():
    """Process user input and generate response"""
    user_input = st.session_state.user_input_widget.strip()
    if not user_input:
        return

    # Content moderation check
    if sensitive_category := contains_sensitive_content(user_input):
        st.warning(f"Content not allowed: {sensitive_category} detected")
        return

    # Emotion analysis
    try:
        emotion_result = st.session_state.emotion_model(user_input)[0]
        st.session_state.conversation.append({
            'role': 'user',
            'content': user_input,
            'emotion': emotion_result['label'],
            'confidence': f"{emotion_result['score']:.0%}"
        })

        # Generate and display response
        with st.spinner("WarmTalk is thinking..."):
            if ai_response := generate_ai_response():
                st.session_state.conversation.append({
                    'role': 'assistant',
                    'content': ai_response
                })

        # Clear input after processing
        st.session_state.user_input_widget = ""
        st.rerun()

    except Exception as e:
        st.error(f"Error processing input: {str(e)}")

# --------- Main Application ---------
def main():
    """Main application layout and logic"""
    initialize_session()

    # --------- Sidebar Settings ---------
    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for AI responses",
            value=st.session_state.api_key
        )

        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()

        st.divider()
        st.caption("About WarmTalk")
        st.markdown("""
        A compassionate AI companion designed
        specifically for elderly users, providing
        warm and supportive conversations.
        """)

    # --------- Main Chat Interface ---------
    st.title("🤗 WarmTalk Companion")
    st.caption("Your gentle digital friend for meaningful conversations")

    # Chat history display
    chat_container = st.container(height=500, border=False)
    with chat_container:
        if not st.session_state.conversation:
            if st.session_state.api_key:
                with st.chat_message("assistant", avatar="🤗"):
                    st.write("Hello dear! I'm WarmTalk. How are you feeling today?")
            else:
                with st.chat_message("assistant", avatar="🔒"):
                    st.write("Please enter your API key to start chatting")

        for msg in st.session_state.conversation:
            avatar = "🧓" if msg['role'] == 'user' else "🤗"
            with st.chat_message(msg['role'], avatar=avatar):
                st.write(msg['content'])
                if msg['role'] == 'user':
                    st.caption(f"Emotion: {msg['emotion']} ({msg['confidence']})")

    # Input area with callback
    st.text_area(
        "Share your thoughts...",
        key="user_input_widget",
        placeholder="Type your message here...",
        on_change=handle_user_input,
        label_visibility="collapsed"
    )

# --------- Application Entry Point ---------
if __name__ == "__main__":
    main()