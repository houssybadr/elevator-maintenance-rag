from fastapi import APIRouter,Request,Depends,HTTPException
from pydantic import BaseModel,Field
from asyncpg import Connection
from loguru import logger
import asyncio
from app.core.security import get_api_key
from app.service.llm import generate
from app.service.prompt_builder import build_prompt
from app.service.embedder import Embedder
from app.service.retriever import retrive
from app.api.dependices import get_connection,get_embedder
from app.api.limiter import limiter

router=APIRouter()
limiter=limiter

class QueryRequest(BaseModel):
    question:str = Field(min_length=10,max_length=500)

@router.post("/query")
@limiter.limit("3/minute")
async def query(
        request: Request,
        query_request:QueryRequest,
        embedder:Embedder=Depends(get_embedder),
        db_connection:Connection=Depends(get_connection),
        api_key:str=Depends(get_api_key)
    ):

    try:
        async with asyncio.timeout(15):
            question=query_request.question
            logger.info("Question recieved")
            embedded_question=str(embedder.embed(question))
            logger.info("Question embedded")
            retrieved_chunks=await retrive(
                db_con=db_connection,
                embedding=embedded_question
                )
            
            if len(retrieved_chunks)==0:
                logger.warning("No chunks found for question")
                return {
                            "answer":"Information not found",
                            "token_usage":0
                        }
            logger.info(f"{len(retrieved_chunks)} Chunks retrieved")

            prompt=build_prompt(
                question=question,
                documents=retrieved_chunks
            )
            response=await generate(prompt)
            logger.info(f"Answer generated | tokens={response[1]}")

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail= "Request timed out"
        )
    return {
        "answer":response[0],
        "token_usage":response[1]
    }