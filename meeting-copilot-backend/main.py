from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
import os
import torchaudio
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import traceback
import time

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
origins = ["http://localhost", "http://localhost:3000", "http://localhost:5173"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
if not HUGGINGFACEHUB_API_TOKEN or not DATABASE_URL:
    raise ValueError("Missing environment variables: HUGGINGFACEHUB_API_TOKEN or DATABASE_URL")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector store
try:
    vector_store = PGVector(connection_string=DATABASE_URL, embedding_function=embeddings, collection_name="meeting_notes")
    print("PGVector initialized successfully.")
except Exception as e:
    print(f"Error initializing PGVector: {e}")
    raise

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.7,
    max_new_tokens=1024,
    timeout=300
)

# Initialize STT pipeline
try:
    stt_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    print("STT pipeline initialized.")
except Exception as e:
    print(f"Warning: STT pipeline failed to initialize: {e}")
    stt_pipeline = None

# Define state for LangGraph
class AgentState(TypedDict):
    query: str
    context: str
    response: str
    copilot_suggestion: str
    external_action: str  # New field for web search or CRM updates

# Define prompt templates
retrieval_prompt = PromptTemplate(input_variables=["query"], template="Retrieve relevant meeting notes for the query: {query}")

analysis_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
You are an AI assistant analyzing a user's query and retrieved context from meeting notes. Determine if the context is sufficient to answer the query directly.

Query: {query}
Context: {context}

Respond with ONLY one of the following words:
'generate_response' - if the context is relevant and sufficient.
'request_clarification' - if the context is empty or irrelevant.
'fetch_external' - if the context is insufficient but a web search or external action could help.

Examples:
Query: What were John's action items?
Context: John to finalize specs by June 10.
Decision: generate_response

Query: What's the latest on project X?
Context: (empty)
Decision: fetch_external

Decision:
"""
)

response_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are an AI meeting co-pilot. Provide a concise, relevant response based on the context and query.

Context: {context}
Query: {query}

Response:
"""
)

copilot_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are an AI meeting co-pilot. Provide ONE concise, actionable suggestion that complements the main response, without repeating it.

Context: {context}
Query: {query}

Suggestion:
"""
)

# LangGraph nodes
def retrieve_context(state: AgentState) -> AgentState:
    start_time = time.time()
    print(f"DEBUG: retrieve_context for query: {state['query']}")
    try:
        query_embedding = embeddings.embed_query(state["query"])
        docs = vector_store.similarity_search_by_vector(query_embedding, k=5)  # Increased k for broader context
        state["context"] = "\n".join([doc.page_content for doc in docs])
        print(f"DEBUG: Retrieved context: {state['context'][:100]}...")
    except Exception as e:
        print(f"ERROR: retrieve_context failed: {e}")
        state["context"] = ""
        state["response"] = "request_clarification"
    print(f"DEBUG: retrieve_context latency: {time.time() - start_time:.2f}s")
    return state

def analyze_query(state: AgentState) -> AgentState:
    start_time = time.time()
    print(f"DEBUG: analyze_query for query: {state['query']}")
    prompt = analysis_prompt.format(query=state["query"], context=state["context"])
    raw_decision = llm.invoke(prompt).strip()
    decision = "request_clarification"
    if "generate_response" in raw_decision:
        decision = "generate_response"
    elif "fetch_external" in raw_decision:
        decision = "fetch_external"
    print(f"DEBUG: Analysis decision: {decision}")
    state["response"] = decision
    print(f"DEBUG: analyze_query latency: {time.time() - start_time:.2f}s")
    return state

def fetch_external(state: AgentState) -> AgentState:
    start_time = time.time()
    print(f"DEBUG: fetch_external for query: {state['query']}")
    # Mock web search or external action
    state["context"] += f"\n[Mock Web Search] Additional info for '{state['query']}': Relevant data fetched from external source."
    state["external_action"] = "Performed mock web search."
    print(f"DEBUG: fetch_external latency: {time.time() - start_time:.2f}s")
    return state

def update_crm(state: AgentState) -> AgentState:
    start_time = time.time()
    print(f"DEBUG: update_crm for query: {state['query']}")
    # Mock CRM update
    state["external_action"] = state.get("external_action", "") + " Updated CRM with meeting action items."
    print(f"DEBUG: update_crm latency: {time.time() - start_time:.2f}s")
    return state

def generate_response(state: AgentState) -> AgentState:
    start_time = time.time()
    print(f"DEBUG: generate_response for query: {state['query']}")
    prompt = response_prompt.format(context=state["context"], query=state["query"])
    raw_response = llm.invoke(prompt)
    cleaned_response = raw_response.split("Response:", 1)[-1].strip() if "Response:" in raw_response else raw_response.strip()
    state["response"] = cleaned_response
    print(f"DEBUG: generate_response latency: {time.time() - start_time:.2f}s")
    return state

def generate_copilot_suggestion(state: AgentState) -> AgentState:
    start_time = time.time()
    print(f"DEBUG: generate_copilot_suggestion for query: {state['query']}")
    prompt = copilot_prompt.format(context=state["context"], query=state["query"])
    raw_suggestion = llm.invoke(prompt)
    cleaned_suggestion = raw_suggestion.split("Suggestion:", 1)[-1].strip() if "Suggestion:" in raw_suggestion else raw_suggestion.strip()
    state["copilot_suggestion"] = cleaned_suggestion
    print(f"DEBUG: generate_copilot_suggestion latency: {time.time() - start_time:.2f}s")
    return state

# Define LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("fetch_external", fetch_external)
workflow.add_node("update_crm", update_crm)
workflow.add_node("generate_response", generate_response)
workflow.add_node("generate_copilot_suggestion", generate_copilot_suggestion)

# Define edges
workflow.add_edge("retrieve_context", "analyze_query")
workflow.add_conditional_edges(
    "analyze_query",
    lambda state: state["response"],
    {
        "generate_response": "generate_response",
        "fetch_external": "fetch_external",
        "request_clarification": END
    }
)
workflow.add_edge("fetch_external", "generate_response")
workflow.add_edge("generate_response", "update_crm")
workflow.add_edge("update_crm", "generate_copilot_suggestion")
workflow.add_edge("generate_copilot_suggestion", END)
workflow.set_entry_point("retrieve_context")
agent = workflow.compile()

# Pydantic models
class QueryInput(BaseModel):
    query: str

# API endpoints
@app.post("/stt")
async def stt_query(input: QueryInput):
    start_time = time.time()
    print("API: /stt endpoint hit.")
    try:
        initial_state = AgentState(query=input.query, context="", response="", copilot_suggestion="", external_action="")
        result = agent.invoke(initial_state)
        response_text = "Could you clarify the query?" if result.get("response") == "request_clarification" else result.get("response", "No response generated.")
        print(f"API: /stt latency: {time.time() - start_time:.2f}s")
        return {
            "query": input.query,
            "response": response_text,
            "copilot_suggestion": result.get("copilot_suggestion", "No suggestion available."),
            "retrieved_context": result.get("context", "No context retrieved."),
            "external_action": result.get("external_action", "No external action performed.")
        }
    except Exception as e:
        print(f"API ERROR in /stt: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/stt_audio")
async def stt_audio(file: UploadFile = File(...)):
    start_time = time.time()
    print("API: /stt_audio endpoint hit.")
    if not stt_pipeline:
        raise HTTPException(status_code=500, detail="STT pipeline not initialized.")
    try:
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        # Normalize audio
        waveform = waveform / waveform.abs().max()
        transcription = stt_pipeline(waveform.squeeze().numpy())["text"]
        os.remove(audio_path)
        print(f"DEBUG: Audio transcribed: {transcription}")
        initial_state = AgentState(query=transcription, context="", response="", copilot_suggestion="", external_action="")
        result = agent.invoke(initial_state)
        response_text = "Could you clarify the audio query?" if result.get("response") == "request_clarification" else result.get("response", "No response generated.")
        print(f"API: /stt_audio latency: {time.time() - start_time:.2f}s")
        return {
            "transcription": transcription,
            "response": response_text,
            "copilot_suggestion": result.get("copilot_suggestion", "No suggestion available."),
            "retrieved_context": result.get("context", "No context retrieved."),
            "external_action": result.get("external_action", "No external action performed.")
        }
    except Exception as e:
        print(f"API ERROR in /stt_audio: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/copilot")
async def copilot_query(input: QueryInput):
    start_time = time.time()
    print("API: /copilot endpoint hit.")
    try:
        initial_state = AgentState(query=input.query, context="", response="", copilot_suggestion="", external_action="")
        result = agent.invoke(initial_state)
        response_text = "More details needed for suggestion." if result.get("response") == "request_clarification" else result.get("response", "No response generated.")
        print(f"API: /copilot latency: {time.time() - start_time:.2f}s")
        return {
            "query": input.query,
            "response": response_text,
            "copilot_suggestion": result.get("copilot_suggestion", "No suggestion available."),
            "retrieved_context": result.get("context", "No context retrieved."),
            "external_action": result.get("external_action", "No external action performed.")
        }
    except Exception as e:
        print(f"API ERROR in /copilot: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/rag")
async def rag_query(input: QueryInput):
    start_time = time.time()
    print("API: /rag endpoint hit.")
    try:
        initial_state = AgentState(query=input.query, context="", response="", copilot_suggestion="", external_action="")
        result = agent.invoke(initial_state)
        response_text = "RAG query needs clarification." if result.get("response") == "request_clarification" else result.get("response", "No RAG answer generated.")
        print(f"API: /rag latency: {time.time() - start_time:.2f}s")
        return {
            "query": input.query,
            "response": response_text,
            "copilot_suggestion": result.get("copilot_suggestion", "No suggestion available."),
            "retrieved_context": result.get("context", "No context retrieved."),
            "external_action": result.get("external_action", "No external action performed.")
        }
    except Exception as e:
        print(f"API ERROR in /rag: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "LangGraph Agent with STT and Co-pilot for AI Meeting Co-pilot"}