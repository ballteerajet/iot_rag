from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from firebase_client import get_sensor_data
from rag_engine import RAGEngine

app = FastAPI()
templates = Jinja2Templates(directory="templates")

rag = RAGEngine()

# โหลดข้อมูลครั้งเดียวตอน start
sensor_data = get_sensor_data()
texts = rag.convert_to_text(sensor_data)
rag.build_vector_store(texts)


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    user_message = data.get("message")

    context_docs = rag.retrieve(user_message)
    answer = rag.generate_answer(user_message, context_docs)

    return {"answer": answer}