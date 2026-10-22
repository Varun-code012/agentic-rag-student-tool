import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from crewai import Agent, Task, Crew, LLM
# ...rest of your existing imports

import litellm

# --- Temporary workaround for a known CrewAI + Groq compatibility bug ---
# CrewAI injects a 'cache_breakpoint' key into messages for prompt-caching
# support, but Groq's API strictly rejects unrecognized fields. This strips
# that key before the request reaches Groq. Remove this once CrewAI ships
# a proper fix for this (tracked upstream: PR #7176).
_original_completion = litellm.completion

def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        for msg in kwargs["messages"]:
            msg.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)

litellm.completion = _patched_completion
# --- end workaround ---

# src/web_agent.py
import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from ddgs import DDGS


@tool("DuckDuckGo Lookup")  # renamed from "Web Search" to avoid colliding with# gpt-oss's built-in 'web_search' tool identifier
                             
def web_search_tool(query: str) -> str:
    """Searches the web and returns a summary of top results for the given query."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)


# Groq, via LiteLLM's provider/model string format — this is what tells
# CrewAI to use Groq's API instead of defaulting to OpenAI.
_groq_llm = LLM(
    model="openai/qwen/qwen3.8-27b",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def search_web(question: str) -> str:
    researcher = Agent(
        role="Web Researcher",
        goal="Find accurate, relevant information to answer the user's question",
        backstory=(
            "You are a careful research assistant who searches the web "
            "and summarizes only what's directly relevant to the question."
        ),
        tools=[web_search_tool],
        llm=_groq_llm,
        verbose=False,
        max_iter=3,  #caps how many times the agent can call its tools before returning an answer
    )

    task = Task(
        description=f"Search the web and find information to answer this question: {question}",
        expected_output="A concise, factual summary that answers the question, based on web search results.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)