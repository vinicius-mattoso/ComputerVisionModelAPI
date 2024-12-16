# app/main.py

from fastapi import FastAPI
from app.routes.classification import router as classification_router
from app.routes.model import router as model_router

app = FastAPI()

# Incluindo os roteadores nas rotas da API
app.include_router(classification_router)
app.include_router(model_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Fashion MNIST API!"}
