from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph import StateGraph, END
from typing import TypedDict, List
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Mock lead capture function
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")

# Load knowledge base
with open('knowledge_base.json', 'r') as f:
    kb = json.load(f)

# Convert to documents
documents = []
for section, content in kb.items():
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(value, dict):
                content_str = ", ".join([f"{k}: {v}" for k, v in value.items()])
            else:
                content_str = str(value)
            doc = Document(page_content=f"{section} - {key}: {content_str}", metadata={"section": section, "key": key})
            documents.append(doc)
    else:
        doc = Document(page_content=f"{section}: {content}", metadata={"section": section})
        documents.append(doc)

# Split and embed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# State
class AgentState(TypedDict):
    messages: List[str]
    intent: str
    user_info: dict
    awaiting: str  # what we're waiting for: name, email, platform

# Nodes
def classify_intent(state):
    messages = state['messages']
    last_msg = messages[-1]
    prompt = f"Classify the user's intent into one of: greet (casual greeting), inquire (product or pricing inquiry), lead (high-intent lead). Message: {last_msg}. Respond with only: greet, inquire, or lead."
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    return {"intent": intent}

def handle_greeting(state):
    response = "Hello! How can I help you with AutoStream today?"
    return {"messages": state['messages'] + [response]}

def handle_inquiry(state):
    messages = state['messages']
    last_msg = messages[-1]
    docs = retriever.get_relevant_documents(last_msg)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question using the context. Be helpful and concise.\nContext: {context}\nQuestion: {last_msg}"
    response = llm.invoke(prompt)
    return {"messages": state['messages'] + [response.content]}

def handle_high_intent(state):
    user_info = state.get('user_info', {})
    awaiting = state.get('awaiting', '')
    if not awaiting:
        response = "Great! To get you started, could you please provide your name?"
        awaiting = "name"
    elif awaiting == "name":
        user_info['name'] = state['messages'][-1]
        response = "Thanks! What's your email address?"
        awaiting = "email"
    elif awaiting == "email":
        user_info['email'] = state['messages'][-1]
        response = "Perfect! What platform do you create content on (e.g., YouTube, Instagram)?"
        awaiting = "platform"
    elif awaiting == "platform":
        user_info['platform'] = state['messages'][-1]
        mock_lead_capture(user_info['name'], user_info['email'], user_info['platform'])
        response = "Thank you! Your lead has been captured. We'll be in touch soon!"
        awaiting = ""
    return {"messages": state['messages'] + [response], "awaiting": awaiting, "user_info": user_info}

# Graph
graph = StateGraph(AgentState)

graph.add_node("classify", classify_intent)
graph.add_node("greet", handle_greeting)
graph.add_node("inquire", handle_inquiry)
graph.add_node("lead", handle_high_intent)
graph.add_node("router", lambda state: state)

graph.set_entry_point("router")

def router_logic(state):
    if state.get('awaiting'):
        return "lead"
    else:
        return "classify"

graph.add_conditional_edges("router", router_logic)

graph.add_conditional_edges("classify", lambda x: x['intent'])

graph.add_edge("greet", END)
graph.add_edge("inquire", END)
graph.add_edge("lead", "router")

app = graph.compile()

# Run conversation
def run_agent():
    state = {"messages": [], "intent": "", "user_info": {}, "awaiting": ""}
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        state['messages'].append(user_input)
        result = app.invoke(state)
        print("Agent:", result['messages'][-1])
        state = result

if __name__ == "__main__":
    run_agent()