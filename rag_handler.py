# rag_handler.py
import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# --- 更新匯入路徑 ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "db"
KNOWLEDGE_BASE_DIR = "knowledge_base"

# 檢查知識庫目錄是否存在
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    os.makedirs(KNOWLEDGE_BASE_DIR)
    print(f"建立知識庫目錄: {KNOWLEDGE_BASE_DIR}，請將您的文件 (pdf, txt) 放入此目錄。")


class RAGHandler:
    def __init__(self):
        # 使用 HuggingFace 上的開源模型進行 embedding
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        # --- 更新：建立 ChromaDB 客戶端並關閉遙測 ---
        chroma_client = chromadb.PersistentClient(path=DB_DIR, settings=chromadb.Settings(anonymized_telemetry=False))

        # 使用新的 Chroma 類別
        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=self.embedding_function,
        )

        # 檢查資料庫是否為空，如果為空則觸發 reindex
        if self.vectorstore._collection.count() == 0:
            print("向量資料庫為空，正在建立新的索引...")
            self.reindex()
        else:
            print("正在載入已存在的向量資料庫...")

    def reindex(self):
        """重新索引所有在 knowledge_base 目錄下的文件"""
        docs = []
        for filename in os.listdir(KNOWLEDGE_BASE_DIR):
            filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding='utf-8') 
                docs.extend(loader.load())

        if not docs:
            print("知識庫目錄為空，無法建立索引。")
            return

        chunks = self.text_splitter.split_documents(docs)
        # 將文件塊添加到現有的 vectorstore 中
        self.vectorstore.add_documents(documents=chunks)
        print(f"索引建立完畢，共處理 {len(docs)} 個文件，生成 {len(chunks)} 個文字塊。")

    def search(self, query, k=3):
        """在向量資料庫中搜尋相關內容"""
        if not self.vectorstore:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

# 注意：我們將不在這裡建立單例，而是在 FastAPI 的啟動事件中建立
# rag_handler = RAGHandler()