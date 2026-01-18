from fastapi import FastAPI

app=FastAPI(title="Rain Predictor API")

@app.get("/")
def home():
    return{"Wellcome to Rain Predictor API"}

@app.get("/health")
def health():
    return{"status":"ok"}