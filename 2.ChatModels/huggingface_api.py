import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

api_token = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    or os.getenv("HF_TOKEN")
)

if not api_token:
    raise ValueError(
        "Hugging Face token not found. Add HUGGINGFACEHUB_API_TOKEN=hf_... "
        "to the project .env file."
    )

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=api_token,
)

model = ChatHuggingFace(llm=llm, max_tokens=100)

result = model.invoke("What is the capital of India")

print(result.content)
