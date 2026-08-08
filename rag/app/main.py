import asyncio
from contextlib import asynccontextmanager
from urllib.error import URLError
from urllib.request import urlopen

from app.api.limiter import limiter
from app.api.routes.query import router
from app.core.config import settings
from app.database.postgres_connection import get_pool
from app.service.embedder import Embedder
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


@asynccontextmanager
async def lifespan(app:FastAPI):
	print("API Sarting ...")
	app.state.pool=await get_pool()
	logger.info("DB Pool initiated")
	app.state.embedder=Embedder()
	app.state.embedder.load()
	logger.info("Embedding model loaded")
	
	#app.state.embedder.load()
	yield
	await app.state.pool.close()
	del app.state.embedder
	print("API  Stopping ...")


app=FastAPI(
	lifespan=lifespan,
	title="Maintenance Manual RAG API",
	description="RAG API for elevator maintenance manual",
	version="1.0.0"
)

# Global exception handler
@app.exception_handler
async def global_exception_handler(request:Request,ex:Exception):
	logger.error(f"Unhandled error : {ex}")

	return JSONResponse(
		status_code=500,
		content={
			"detail":"Internal server error"
		}
	)

# Adding access rate limiter
app.state.limiter=limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)

# Adding middlewares
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["POST","GET"],
	allow_headers=["X-API-Key","Content-Type"]
)

# Adding routers
app.include_router(router)

async def _check_database(app: FastAPI):
	checks = {}
	async with app.state.pool.acquire() as connection:
		await connection.fetchval("SELECT 1")
		checks["postgres_connection"] = {"ok": True}

		vector_exists = await connection.fetchval(
			"""
			SELECT EXISTS (
				SELECT 1 FROM pg_extension WHERE extname = 'vector'
			)
			"""
		)
		if not vector_exists:
			raise RuntimeError("pgvector extension is not installed.")
		checks["pgvector_extension"] = {"ok": True}

		documents_table_exists = await connection.fetchval(
			"""
			SELECT EXISTS (
				SELECT 1
				FROM information_schema.tables
				WHERE table_schema = 'public'
				AND table_name = 'documents'
			)
			"""
		)
		if not documents_table_exists:
			raise RuntimeError("documents table is missing.")
		checks["documents_table"] = {"ok": True}

	return checks


async def _check_internet_connectivity():
	def _probe():
		with urlopen(
			settings.healthcheck_internet_url,
			timeout=settings.healthcheck_timeout_seconds,
		) as response:
			status = getattr(response, "status", 200)
			if status >= 400:
				raise RuntimeError(f"Internet check returned HTTP {status}.")

	try:
		await asyncio.to_thread(_probe)
		return {"ok": True}
	except (URLError, TimeoutError, OSError) as exc:
		return {"ok": False, "error": str(exc)}


@app.get("/health")
@limiter.limit("10/minute")
async def health(request: Request):
	health_checks = {}
	global_status = "ok"
	http_status = 200

	internet_check = await _check_internet_connectivity()
	health_checks["internet_connectivity"] = internet_check
	if not internet_check["ok"]:
		global_status = "degraded"
		http_status = 503

	try:
		health_checks.update(await _check_database(request.app))
	except Exception as exc:
		health_checks["database"] = {"ok": False, "error": str(exc)}
		global_status = "degraded"
		http_status = 503

	embedder = getattr(request.app.state, "embedder", None)
	embedder_ready = embedder is not None and getattr(embedder, "model", None) is not None
	health_checks["embedder_model_loaded"] = {"ok": embedder_ready}
	if not embedder_ready:
		global_status = "degraded"
		http_status = 503

	return JSONResponse(
		status_code=http_status,
		content={
			"status": global_status,
			"checks": health_checks,
		},
	)
