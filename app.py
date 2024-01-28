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

with st.sidebar:   
    
    cohere_key = st.text_input("Enter Cohere API key", type="password")
    os.environ["COHERE_API_KEY"] = cohere_key
    gemini_key = st.text_input("Enter Google Gemini API key", type="password")
    
    pinecone_key = st.text_input("Enter Pinecone API key", type="password")
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

def card(thubmnail, title, url, context):
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
            <div class="col-md-4 col-sm-4">
                 <div class="position-relative">
                     <a href={url}><img src={thubmnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                 </div>
             </div>
             <div  class="col-md-8 col-sm-8">
                 <a href={url}>{title}</a>
                 <br>
                 <span style="color: #808080;">
                     <small>{context[:200].capitalize()+"...."}</small>
                 </span>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)

    
st.write("""
# YouTube Q&A
Ask me a question!
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

query = st.text_input("Search!", "")

if query != "":
    # if not cohere_key:
    #     st.info("Please enter your Cohere API key")
    #     st.stop()
    if not gemini_key:
        st.info("Please enter your Google Gemini API key")
        st.stop()
    # if not pinecone_key:
    #     st.info("Please enter your Pinecone API key")
    #     st.stop()
    encoder = CohereEncoder()
    rl = RouteLayer(encoder=encoder, routes=routes)
    Routelay = rl(query)
    if Routelay.name == "politics":
        st.write("I cannot talk about politics/chitchat")
    else:
        xq = retriever.encode([query]).tolist()
        xc = index.query(vector=xq, top_k=5, include_metadata=True)
        if xc['matches'][0]['score'] < 0.5:
            st.write("I do not have knowledge about this topic")
        else:
            for context in xc['matches']:
                card(
                    context['metadata']['thumbnail'],
                    context['metadata']['title'],
                    context['metadata']['url'],
                    context['metadata']['text']
            )
