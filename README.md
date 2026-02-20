🚀 Agentic RAG Assistant with Web Search & Tool Integration

An intelligent multi-tool AI assistant that combines Retrieval-Augmented Generation (RAG), web search, and mathematical reasoning into a single agent capable of dynamic decision-making.

🧠 Overview

This project implements an agentic AI system that can:

Retrieve company-specific knowledge using RAG

Fetch real-time information from the web

Perform mathematical computations

Dynamically decide which tool to use

Unlike traditional RAG systems, this assistant reasons and selects tools autonomously, making it more powerful and flexible.

⚙️ Architecture
User Query
     ↓
Agent (Decision Maker)
     ↓
 ┌───────────────┬───────────────┬───────────────┐
 │               │               │               │
RAG Tool     Tavily Tool     Math Tool      LLM Direct
 │               │               │               │
 └───────────────┴───────────────┴───────────────┘
                     ↓
              Final Response
🧰 Tools Used
📄 1. RAG Tool

Retrieves internal company documents (Terms, Salary, Policies)

Uses vector search (Chroma)

🌐 2. Web Search Tool

Powered by Tavily API

Fetches real-time and external knowledge

🧮 3. Math Tool

Handles arithmetic operations

Used for salary calculations and numeric queries

🔄 Agent Workflow

Receive user query

Analyze intent

Select appropriate tool(s):

Company-related → RAG

Calculation → Math Tool

General/latest info → Web Search

Combine outputs

Generate final response

🧪 Example Queries
✅ Company Query

Input:

What is the fresher salary?

Output:
→ ₹20,000 per month

✅ Calculation Query

Input:

Calculate yearly salary for fresher

Agent Steps:

Retrieve salary from RAG

Calculate using Math Tool

Output:
→ ₹2.4 LPA

✅ Web Query

Input:

Latest AI trends in 2026

Output:
→ Retrieved using Tavily API

🔥 Key Features

✅ Agent-based decision making

✅ Multi-tool integration

✅ Hybrid knowledge (internal + external)

✅ Real-world use case (company policy assistant)

✅ Modular and extensible design
