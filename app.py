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
st.markdown("""
<style>
    body {
        background-color: white !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

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
        "Where do you want to go today?",
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
    background_color = "#ffecf2" if is_even else "white"  # Alternating light grey and white

    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start mb-3" style="background-color: {background_color} ;">
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
import streamlit as st
st.markdown("<h1 style='text-align: center;'>Semantica Vision</h1>", unsafe_allow_html=True)
st.write("""
### Semantic routing based Chatbot and Video Search
""")

# Description section
st.write("""###### It has knowledge(limited) about Artificial Intelligence/Machine Learning/Communications  
    Note: 
    1) Multiple concurrent requests on multiple devices will lead to error. So try again later if that so 
    2) No. of requests per minute are limited and hence that can lead to error too. So try again later if that so 
    Reason: Free Trial""")

# Add some space between sections
st.write("USE CASE 1 - Out of Context Queries - Will say that it doesn't know eg. How to make a salad?")
st.write("USE CASE 2 - In context Queries - Will summarize and suggest videos with videos starting at the exact timestamp where it is related to the query")
st.write("USE CASE 3 - In Context but restricted topics - currently it supports - Politics, chitchat and personal opinions - will respond that it cannot talk about it")  # Empty line for spacing

# Ask me a question section
st.write("### Ask me a question!")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)
st.markdown("""
<style>
    input[type="text"] {
        background-color: #ffecf2;  /* Pink */
        border: 1px solid #FFB6C1;
        color: black !important;  /* Adjust text color as needed */
        padding: 8px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
query = st.text_input("Search!", "")



## Define Your Prompt
prompt=[
    """
    You are an expert in the knowledge of Artificial Intelligence, Machine Learning and Communications. Summarize your answer in no more than 250 characters.

    """
]
def out(response,summary):
    custom_css = """
    <style>
        .streamlit-markdown-container {
            color: black !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Box styling
    box_color = "#ffecf2"
    if summary == False:
        # box_color = "#ffecf2"
        # colored_box = f'<div style="background-color:{box_color}; padding:10px; border-radius:5px;"><b>{response}</div>'
        # st.markdown(colored_box, unsafe_allow_html=True)
    
        colored_box = f'<div style="background-color:{box_color}; padding:10px; border-radius:5px; color:black;"><b>{response}</b></div>'
        st.markdown(colored_box, unsafe_allow_html=True)
    else:
        colored_box = f'<div style="background-color:{box_color}; padding:10px; border-radius:5px;color:black;"><b>Summary:</b>{response}</b></div>'
        st.markdown(colored_box, unsafe_allow_html=True)
try:        
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
            if Routelay.name == "politics" or Routelay.name == "chitchat":
                #st.write("I cannot talk about politics/chitchat/hate comments/personal opinions")
                out("I cannot talk about politics/chitchat/hate comments/personal opinions",False)
            else:
                xq = retriever.encode([query]).tolist()
                xc = index.query(vector=xq, top_k=5, include_metadata=True)
                if xc['matches'][0]['score'] < 0.5:
                    #st.write("I do not have knowledge about this topic")
                    out("I do not have knowledge about this topic", False)
                else:
                    
                    response=get_gemini_response(query,prompt)
                    #box_color = "#F0FFFF"
                    summary = True
                    out(response, summary)
                    st.write("---------------------------------------------")
                    #st.write(response)
                    is_even = True
                    for context in xc['matches'][:3]:
                        card(
                            context['metadata']['thumbnail'],
                            context['metadata']['title'],
                            context['metadata']['url'],
                            is_even )
                        if is_even == True:
                            is_even = False
                        else:
                            is_even = True
except:
    st.write("Due to multiple requests on multiple devices, try again later")
# Your name
name = "Naman Rajendra Joshi"

# Copyright text
copyright_text = f"© {name} - All Rights Reserved"

# Your Streamlit app content goes here

# Display copyright text at the bottom
st.markdown(
    f'<div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);">{copyright_text}</div>',
    unsafe_allow_html=True
)
