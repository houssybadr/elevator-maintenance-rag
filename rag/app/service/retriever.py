from fastapi import Depends
from asyncpg import Connection
from app.core.config import settings
from app.api.dependices import get_connection
import json


async def retrive(db_con:Connection,embedding:str,top_k:int =None)-> list:
    top_k=top_k if not None else settings.top_k
    rows=await db_con.fetch(
        """
        SELECT  
            content,
            metadata,
            1 - (embedding <=> $1::vector) AS similarity 
        FROM documents
        WHERE 1 - (embedding <=> $1::vector) >= $2  
        ORDER BY similarity DESC
        LIMIT $3
        """,
        embedding,
        settings.min_similarity,
        top_k
    )

    results=[]
    for row in rows:
        row=dict(row)
        row["metadata"]=json.loads(row["metadata"])
        results.append(row)
    return results

