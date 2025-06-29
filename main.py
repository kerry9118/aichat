# main.py (多智能體協作版 - 獨立上下文版)
import asyncio
import re
import base64
import traceback
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

from faster_whisper import WhisperModel
import ollama
import edge_tts
from rag_handler import RAGHandler

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] 伺服器啟動中...")
    try:
        models["stt_model"] = WhisperModel("small", device="cpu", compute_type="int8")
        print("[INFO] Whisper STT 模型已成功載入。")
        models["rag_handler"] = RAGHandler()
        print("[INFO] RAG 處理器已成功初始化。")
    except Exception as e:
        print(f"[ERROR] 模型載入失敗: {e}")
        traceback.print_exc()
    yield
    print("[INFO] 伺服器關閉中...")
    models.clear()
    print("[INFO] 模型資源已清除。")

app = FastAPI(lifespan=lifespan)

TTS_VOICE = "zh-CN-YunxiNeural"

# --- 修正點：將範例 JSON 中的 { 和 } 分別替換為 {{ 和 }} ---
PLANNER_PROMPT = """
你是一個專家級的任務規劃器。你的工作是分析使用者的問題，並決定下一步的行動方案。
你有兩個可用的工具：'Math_Teacher' 和 'Chinese_Teacher'。

你的任務是：
1. 判斷問題應該由哪個工具處理。'Math_Teacher' 用於數學問題，'Chinese_Teacher' 用於其他所有問題。
2. 如果選擇 'Chinese_Teacher'，你必須從問題中提取出最適合用於資料庫搜尋的 1-3 個核心關鍵詞。
3. 你的回應必須是一個結構化的 JSON 物件，其中包含 "tool" 和 "keywords" 兩個鍵。

以下是範例：
使用者問題: "15乘以25等於多少？"
你的 JSON 回應:
{{
  "tool": "Math_Teacher",
  "keywords": ""
}}

使用者問題: "你好，可以為我寫一首關於月亮的詩嗎？"
你的 JSON 回應:
{{
  "tool": "Chinese_Teacher",
  "keywords": "月亮 詩"
}}

使用者問題: "靜夜思是誰寫的？"
你的 JSON 回應:
{{
  "tool": "Chinese_Teacher",
  "keywords": "靜夜思 作者"
}}

現在，請分析以下最新的使用者問題，並只回傳 JSON 格式的回應。
使用者問題: "{user_input}"
"""

CHINESE_TEACHER_PROMPT = """你是一位知識淵博、溫文儒雅的語文老師。你的名字叫李老師。你的任務是：1. 與使用者進行友好對話，回答有關文學、詩歌、歷史等問題。2. 優先使用提供的「參考資料」來回答問題。如果沒有參考資料，則用你自己的知識回答。3. 回答要清晰、簡潔、有條理。"""
CONTEXT_INJECTION_PROMPT = """---這是一些可能相關的參考資料，請優先參考它們來回答最新的使用者問題。如果資料不相關或不足以回答，請使用你自己的知識來回答，但仍然要保持你作為語文老師的身份。參考資料如下：{context}---"""

MATH_TEACHER_PROMPT = """你是一位頂級的數學老師和解題專家。你的任務是清晰、準確、一步步地解決使用者提出的數學問題。請根據對話歷史進行回答。"""

@app.get("/", response_class=HTMLResponse)
async def get_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[INFO] WebSocket 連線已建立。")

    # 為每個專家建立獨立的對話歷史
    agent_histories = {
        "Chinese_Teacher": [{"role": "system", "content": CHINESE_TEACHER_PROMPT}],
        "Math_Teacher": [{"role": "system", "content": MATH_TEACHER_PROMPT}]
    }

    try:
        while True:
            print("\n[LIFECYCLE] 等待前端傳送音訊...")
            audio_base64 = await websocket.receive_text()
            audio_bytes = base64.b64decode(audio_base64)
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_bytes)
            segments, info = models["stt_model"].transcribe("temp_audio.wav", beam_size=5, language="zh")
            user_input = "".join([segment.text for segment in segments])
            print(f"[RESULT] STT 辨識結果: '{user_input}'")

            if not user_input.strip():
                print("[WARN] STT 結果為空，忽略此次請求。")
                continue

            await websocket.send_json({"type": "user_text", "data": user_input})

            # 1. 規劃步驟
            print("\n--- 1. 開始規劃步驟 ---")
            try:
                planner_formatted_prompt = PLANNER_PROMPT.format(user_input=user_input)
            except KeyError as e:
                print(f"[FATAL_ERROR] 格式化 PLANNER_PROMPT 時發生 KeyError: {e}。請檢查 Prompt 中的大括號是否都已正確轉義 ({{ 和 }})。")
                continue

            planner_response = ollama.chat(model='qwen3:8b', messages=[{"role": "user", "content": planner_formatted_prompt}], stream=False)
            plan_json_str = planner_response['message']['content']
            print(f"[DEBUG] 規劃模型原始回覆: '{plan_json_str}'")
            try:
                match = re.search(r'\{.*\}', plan_json_str, re.DOTALL)
                plan = json.loads(match.group())
                chosen_agent = plan.get("tool", "Chinese_Teacher")
                keywords = plan.get("keywords", user_input)
            except (json.JSONDecodeError, AttributeError):
                chosen_agent = "Chinese_Teacher"
                keywords = user_input
            print(f"[RESULT] 規劃結果 -> 工具: '{chosen_agent}', 關鍵詞: '{keywords}'")

            # 將使用者輸入加入到被選中專家的歷史中
            agent_histories[chosen_agent].append({"role": "user", "content": user_input})

            # 2. 呼叫專家智能體 (流式)
            print("\n--- 2. 呼叫專家智能體並流式傳輸回覆 ---")
            full_ai_response_text = ""
            response_stream = None

            if chosen_agent == "Math_Teacher":
                print("[INFO] 執行分支: 呼叫數學老師...")
                response_stream = ollama.chat(model='qwen3:8b', messages=agent_histories["Math_Teacher"], stream=True)
            else:
                print("[INFO] 執行分支: 呼叫語文老師...")
                relevant_docs = models["rag_handler"].search(keywords)
                current_llm_messages = list(agent_histories["Chinese_Teacher"])
                if relevant_docs:
                    print("[INFO] 發現相關知識庫內容，準備注入上下文。")
                    context = "\n\n".join(relevant_docs)
                    context_prompt = CONTEXT_INJECTION_PROMPT.format(context=context)
                    current_llm_messages.insert(-1, {"role": "system", "name":"knowledge_base_context", "content": context_prompt})
                else:
                    print("[INFO] 未發現知識庫內容，進行通用對話。")
                response_stream = ollama.chat(model='qwen3:8b', messages=current_llm_messages, stream=True)

            # 處理流式回覆
            for chunk in response_stream:
                if 'content' in chunk['message']:
                    text_chunk = chunk['message']['content']
                    full_ai_response_text += text_chunk
                    await websocket.send_json({"type": "text_chunk", "data": text_chunk})

            await websocket.send_json({"type": "stream_end"})
            print("[INFO] 文字流傳輸完畢。")

            cleaned_response = re.sub(r'<think>.*?</think>', '', full_ai_response_text, flags=re.DOTALL).strip()
            # 將 AI 回覆加入到被選中專家的歷史中
            agent_histories[chosen_agent].append({"role": "assistant", "content": cleaned_response})
            print(f"[RESULT] 專家回覆 (已清理):\n---\n{cleaned_response}\n---")

            # 3. 文字轉語音 (TTS)
            print("\n--- 3. 開始 TTS 處理 ---")
            try:
                if cleaned_response.strip():
                    communicate = edge_tts.Communicate(cleaned_response, TTS_VOICE)
                    audio_buffer = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio": audio_buffer += chunk["data"]
                    if audio_buffer:
                        await websocket.send_bytes(audio_buffer)
                        print("[INFO] 音訊已成功傳送到前端。")
            except Exception as e:
                print(f"[ERROR] TTS 處理時發生嚴重錯誤: {e}")

    except Exception as e:
        print(f"[ERROR] WebSocket 主循環發生錯誤: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] WebSocket 連線已關閉。")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
