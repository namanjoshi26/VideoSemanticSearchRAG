import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import CohereEncoder
import os
from dotenv import load_dotenv
from getpass import getpass
import logging
import cohere
import google.generativeai as genai
import time

# Configure logger
logging.getLogger("complete").setLevel(logging.WARNING)


# Load environment variables
load_dotenv()


# # Assign credentials from environment variable or streamlit secrets dict
# co = cohere.Client(os.getenv("COHERE_API_KEY")) or st.secrets["COHERE_API_KEY"]

@st.experimental_singleton
def init_pinecone(api):
    pc = Pinecone(api_key=api)
    return pc.Index('youtube-search')
    
@st.experimental_singleton
def init_retriever():
    return SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')
    
def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text
    
cohere_key = st.secrets["COHERE_API_KEY"]
os.environ["COHERE_API_KEY"] = cohere_key
gemini_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = gemini_key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
pinecone_key = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_API_KEY"] = pinecone_key

# with st.sidebar:   
    
#     cohere_key = st.text_input("Enter Cohere API key", type="password")
#     os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
#     gemini_key = st.text_input("Enter Google Gemini API key", type="password")
#     os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
#     genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
#     pinecone_key = st.text_input("Enter Pinecone API key", type="password")
#     pinecone_key = st.secrets["COHERE_API_KEY"]
#     os.environ["PINECONE_API_KEY"] = pinecone_key

# Check if API key is provided
if cohere_key:
    # Assign credentials from the provided API key
    co = cohere.Client(cohere_key)

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president" "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

routes = [politics,chitchat]


    
try:
    index = init_pinecone(pinecone_key)
    
except:
    # Handle the configuration error, e.g., print a user-friendly message
    st.write("Enter PineCone API Key")
    
#index = init_pinecone(pinecone_key)
retriever = init_retriever()

def card(thumbnail, title, url, is_even):
    background_color = "#f8f9fa" if is_even else "white"  # Alternating light grey and white

    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start mb-3" style="background-color: {background_color};">
            <div class="col-md-4 col-sm-4">
                 <div class="position-relative">
                     <a href={url}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                 </div>
             </div>
             <div  class="col-md-8 col-sm-8">
                 <a href={url}>{title}</a>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)

    
# Title section
st.write("""
# RAG based Chatbot and Semantic Video Search without any Hallucinations
""")

# Description section
st.write("###### It has knowledge(limited) about Artificial Intelligence/Machine Learning/Communications")

# Add some space between sections
st.write("")  # Empty line for spacing

# Ask me a question section
st.write("### Ask me a question!")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)
st.markdown("""
<style>
    input[type="text"] {
        background-color: #FFC0CB;  /* Pink */
        border: 1px solid #FFB6C1;
        color: black;  /* Adjust text color as needed */
        padding: 8px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
query = st.text_input("Search!", "")

# st.write("### Ask me a question!")

# # You can also add the search icon using Bootstrap's input group
# query = st.text_input("", value="", placeholder="Search", key="search_input")

# # You can also add the search icon using Bootstrap's input group
# search_icon = '<span class="input-group-append"><span class="input-group-text"><i class="fas fa-search"></i></span></span>'
# st.markdown(f'<div class="input-group">{query}{search_icon}</div>', unsafe_allow_html=True)


## Define Your Prompt
prompt=[
    """
    You are an expert in the knowledge of Artificial Intelligence, Machine Learning and Communications. Summarize your answer in no more than 250 characters.

    """
]
def out(response):
    box_color = "#FFC0CB"
    colored_box = f'<div style="background-color:{box_color}; padding:10px; border-radius:5px;"><b>{response}</div>'
    st.markdown(colored_box, unsafe_allow_html=True)
if query != "":
    with st.spinner("Processing..."):
        if not cohere_key:
            st.info("Please enter your Cohere API key")
            st.stop()
        if not gemini_key:
            st.info("Please enter your Google Gemini API key")
            st.stop()
        if not pinecone_key:
            st.info("Please enter your Pinecone API key")
            st.stop()
        encoder = CohereEncoder()
        rl = RouteLayer(encoder=encoder, routes=routes)
        Routelay = rl(query)
        if Routelay.name == "politics":
            #st.write("I cannot talk about politics/chitchat/hate comments/personal opinions")
            out("I cannot talk about politics/chitchat/hate comments/personal opinions")
        else:
            xq = retriever.encode([query]).tolist()
            xc = index.query(vector=xq, top_k=5, include_metadata=True)
            if xc['matches'][0]['score'] < 0.5:
                #st.write("I do not have knowledge about this topic")
                out("I do not have knowledge about this topic")
            else:
                
                response=get_gemini_response(query,prompt)
                box_color = "#F0FFFF"
                colored_box = f'<div style="background-color:{box_color}; padding:10px; border-radius:5px;"><b>Summary:</b> {response}</div>'
                st.markdown(colored_box, unsafe_allow_html=True)
                st.write("---------------------------------------------")
                #st.write(response)
                is_even = True
                for context in xc['matches']:
                    card(
                        context['metadata']['thumbnail'],
                        context['metadata']['title'],
                        context['metadata']['url'],
                        is_even )
                    if is_even == True:
                        is_even = False
                    else:
                        is_even = True
