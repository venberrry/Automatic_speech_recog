from fastapi import FastAPI
from endpoints import emotion

app = FastAPI()

app.include_router(emotion.router)

