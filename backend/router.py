import requests

def build_prompt(context, question, level):

    if level == "beginner":
        return f"""
You are a strict educational instructor.

Rules:
- Answer ONLY using the provided context.
- If answer is not in context, say:
  "This topic is not covered in this module."
- Explain step-by-step.
- Keep explanation simple.

Context:
{context}

Question:
{question}
"""

    else:
        return f"""
You are a senior mentor.

Rules:
- Prefer using the provided context.
- You may expand slightly for clarity.
- Provide structured explanation.

Context:
{context}

Question:
{question}
"""

def call_llm(prompt):

    response = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
	    "keep_alive": "10m"
        }
    )

    data = response.json()

    if "response" in data:
        return data["response"]
    else:
        return "Error: LLM did not return expected response."
