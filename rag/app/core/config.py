from pydantic_settings import BaseSettings

class Settings(BaseSettings):
	# Postgres
	postgres_host:str
	postgres_port: int = 5432
	postgres_db:str
	postgres_user:str
	postgres_password:str
	
	# Llm
	google_api_key:str
	groq_api_key:str

	# Security
	api_key:str

	# Model
	embedding_model:str = "BAAI/bge-m3"
	max_tokens:int = 512
	top_k:int=7
	min_similarity:float=.7
	mock_embedding_dim:int = 1024
	skip_embedder_load:bool = False
	healthcheck_internet_url:str = "https://www.google.com/generate_204"
	healthcheck_timeout_seconds:float = 3.0

	class Config:
		env_file="../.env"

settings=Settings()