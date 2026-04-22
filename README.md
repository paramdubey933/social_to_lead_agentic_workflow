# AutoStream Conversational AI Agent

This project implements a Conversational AI Agent for AutoStream, a fictional SaaS company providing automated video editing tools for content creators. The agent handles user intents, retrieves knowledge using RAG, and captures leads when users show high intent.

## Features

- **Intent Identification**: Classifies user messages into casual greeting, product/pricing inquiry, or high-intent lead.
- **RAG-Powered Knowledge Retrieval**: Uses a local knowledge base stored in `knowledge_base.json` to answer questions about pricing, features, and policies.
- **Lead Capture**: Collects user information (name, email, platform) and calls a mock API when high intent is detected.

## How to Run the Project Locally

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/social_to_lead_agentic_workflow.git
   cd social_to_lead_agentic_workflow
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   python agent.py
   ```

5. Interact with the agent by typing messages. Type 'exit' to quit.

## Architecture Explanation

This agent is built using LangGraph, a framework for building stateful, multi-actor applications with LLMs. LangGraph was chosen over AutoGen because it provides better control over conversation flow and state management through its graph-based structure, allowing for complex conditional routing based on user intent and conversation state.

The architecture consists of several nodes in a state graph:
- **Router**: Checks if the agent is awaiting user information for lead capture.
- **Classify**: Uses GPT-4o-mini to classify user intent.
- **Greet**: Handles casual greetings.
- **Inquire**: Performs RAG retrieval to answer product/pricing questions.
- **Lead**: Manages the lead capture process, asking for name, email, and platform sequentially.

State is managed using a TypedDict that includes conversation messages, current intent, user information, and what the agent is awaiting. This allows the agent to maintain context across 5-6 conversation turns, remembering previous interactions and collected data.

For RAG, the knowledge base is stored as JSON, converted to documents, split, embedded using OpenAI embeddings, and stored in a FAISS vector store for efficient retrieval.

## WhatsApp Integration

To integrate this agent with WhatsApp, I would use WhatsApp Business API with webhooks:

1. Set up a WhatsApp Business account and obtain API credentials.
2. Create a webhook endpoint (e.g., using Flask or FastAPI) that receives messages from WhatsApp.
3. When a message is received, pass it to the LangGraph agent for processing.
4. Send the agent's response back to WhatsApp via the API.
5. Handle media and other message types as needed.
6. Ensure compliance with WhatsApp's policies and rate limits.

The webhook would essentially wrap the `run_agent` function, adapting it to handle WhatsApp's message format instead of console input.