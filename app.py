import streamlit as st
from transformers import pipeline
# ---------------- PAGE CONFIG ---------------- #


st.set_page_config(
    page_title="GPT-2 Chat AI",
    page_icon="🤖",
    layout="wide"
)


# ---------------- CUSTOM CSS ---------------- #

st.markdown(
"""
<style>

.stApp {
    background-color: #0e1117;
}


.title {
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
}


.subtitle{
    text-align:center;
    color:#8b949e;
}


.user-message {
    background-color:#2f3336;
    color:white;
    padding:15px;
    border-radius:15px;
    margin:10px;
    text-align:right;
}


.bot-message {
    background-color:#1f6feb;
    color:white;
    padding:15px;
    border-radius:15px;
    margin:10px;
}


</style>

""",
unsafe_allow_html=True
)



# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🤖 GPT-2 AI")

    st.write(
        "Prodigy GenAI Internship"
    )

    st.divider()

    st.write("Model:")
    st.success("GPT-2 Transformer")

    st.write("Developer:")
    st.info("SaiKishore P")



# ---------------- HEADER ---------------- #

st.markdown(
    "<div class='title'>ChatGPT Style GPT-2 Bot 🤖</div>",
    unsafe_allow_html=True
)


st.markdown(
    "<div class='subtitle'>Ask anything and let AI complete your thoughts ✨</div>",
    unsafe_allow_html=True
)



# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():

    generator = pipeline(
        "text-generation",
        model="gpt2-medium"
    )

    return generator



generator = load_model()



# ---------------- CHAT MEMORY ---------------- #

if "messages" not in st.session_state:

    st.session_state.messages=[]



# DISPLAY OLD CHATS

for message in st.session_state.messages:


    if message["role"]=="user":

        st.markdown(
            f"""
            <div class='user-message'>
            🧑 {message['content']}
            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            f"""
            <div class='bot-message'>
            🤖 {message['content']}
            </div>
            """,
            unsafe_allow_html=True
        )



# ---------------- INPUT ---------------- #

prompt = st.chat_input(
    "Message GPT-2..."
)



if prompt:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )


    with st.spinner(
        "GPT-2 is thinking..."
    ):


        result = generator(

            prompt,

            max_length=120,

            temperature=0.7,

            top_p=0.9,

            repetition_penalty=1.2,

            do_sample=True

        )


        response=result[0]["generated_text"]



    st.session_state.messages.append(

        {
            "role":"assistant",
            "content":response
        }

    )


    st.rerun()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
