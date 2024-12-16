# app/routes/classification.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import json

router = APIRouter()

@router.get("/classification-report/")
async def get_classification_report():
    report_json_path = "app/models/classification_report.json"
    
    # Carregar o relatório do arquivo JSON
    with open(report_json_path, 'r') as f:
        report = json.load(f)
    
    return JSONResponse(content=report)
