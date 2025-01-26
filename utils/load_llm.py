from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def load_llama():
    llm = LlamaCpp(
        model_path="./models/llama/llama3.gguf",
        temperature=0.1,
        n_ctx=2048,
        n_batch=2048,
        
        n_gpu_layers=33,
        callback_manager=callback_manager,
        verbose=True,
        use_mlock=True,
        f16_kv=True,
        stream=True,
        max_tokens=70,
    )
    return llm
