from fastapi import APIRouter
from pydantic import BaseModel
from utils.get_answer import get_answer
from utils.load_llm import load_llama


router = APIRouter()

llm = load_llama()

class QueryRequest(BaseModel):
    query: str

@router.post("/get_answer")
async def example_endpoint(request: QueryRequest):
    query = request.query  
    answer = get_answer(query, llm) 
    return {"answer": answer}  
