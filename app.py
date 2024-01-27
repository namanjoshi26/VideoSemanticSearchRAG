import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from semantic_router import Route
from semantic_router.layer import RouteLayer

@st.experimental_singleton
def init_pinecone():
    pc = Pinecone(api_key="b548349d-858c-44cd-8e13-b56ad80eab3e")
    return pc.Index('youtube-search')
    
@st.experimental_singleton
def init_retriever():
    return SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

with st.sidebar:    
    cohere_key = st.text_input("Enter Cohere API key", type="password")
    gemini_key = st.text_input("Enter Google Gemini API key", type="password")

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


    

index = init_pinecone()
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
    if not cohere_key:
        st.info("Please enter your OpenAI API key")
        st.stop()
    if not gemini_key:
        st.info("Please enter your Googel Gemini API key")
        st.stop()
    encoder = CohereEncoder()
    rl = RouteLayer(encoder=encoder, routes=routes)
    RouteLayer = rl(query)
    if RouteLayer == "politics":
        print("I don't have knowledge about this topic")
        break
        
    xq = retriever.encode([query]).tolist()
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    
    for context in xc['matches']:
        card(
            context['metadata']['thumbnail'],
            context['metadata']['title'],
            context['metadata']['url'],
            context['metadata']['text']
        )
