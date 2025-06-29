# test_server.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    print(">>> 請求已成功到達！<<<")
    return {"message": "測試伺服器運作正常！"}

if __name__ == "__main__":
    print("--- 正在 127.0.0.1:8000 上啟動一個最簡單的測試伺服器 ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)