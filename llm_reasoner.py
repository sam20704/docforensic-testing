# services/reasoning/llm_reasoner.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

from digitrase_vision.services.reasoning.prompt_builder import build_reasoning_prompt

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

TEMPERATURE = 0.1
MAX_TOKENS = 250


def generate_reasoning(validated_signals):

    prompt = build_reasoning_prompt(validated_signals)

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": "You are a digital document forensic analyst."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
