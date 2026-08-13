import os
import json
import logging
import torch
from typing import List
from dotenv import load_dotenv, find_dotenv

# ==========================================
# THƯ VIỆN LANGCHAIN & HUGGINGFACE
# ==========================================
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
load_dotenv(find_dotenv())
# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Cấu hình Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # Thay bằng mật khẩu Neo4j của bạn

# Cấu hình Đường dẫn Model Local
PHOBERT_PATH = r"C:\phobert_medical"  # Đường dẫn tới thư mục PhoBERT của bạn


# ==========================================
# 1. LỚP TÙY CHỈNH: PHOBERT EMBEDDINGS (LOCAL)
# ==========================================
class PhoBERTEmbeddings(Embeddings):
    """Wrapper tùy chỉnh để LangChain sử dụng PhoBERT chạy Local"""
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Đang tải mô hình PhoBERT lên thiết bị: {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        
        # Tiền xử lý tiếng Việt chuyên ngành bằng pyvi
        text_segmented = [ViTokenizer.tokenize(text) for text in texts]
        
        inputs = self.tokenizer(
            text_segmented, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Lấy vector [CLS] đại diện ngữ nghĩa cho toàn bộ chunk văn bản
        embeddings = outputs.last_hidden_state[:, 0, :].tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ==========================================
# 2. HỆ THỐNG GRAPH RAG CHÍNH
# ==========================================
class MedicalGraphRAG:
    def __init__(self):
        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("[*] Đang khởi tạo hệ thống Medical GraphRAG...")
        
        # 1. Load Local Embedding Model
        self.embeddings = PhoBERTEmbeddings(model_path=PHOBERT_PATH)
        
        # 2. Khởi tạo LLM (Temperature 0.1 để đảm bảo tính chính xác y khoa)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            temperature=0,  
            max_output_tokens=1024
        )
        
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None

    # -----------------------------------------------------------------------------------
    # LUỒNG 1: BỘ ĐỊNH TUYẾN Ý ĐỊNH (INTENT ROUTER)
    # -----------------------------------------------------------------------------------
    def analyze_user_query(self, user_query: str) -> str:
        """Phân tích ý định để quyết định sử dụng Luồng Giao tiếp hay Luồng Y tế"""
        system_prompt = f"""Phân loại câu hỏi của người dùng và trả về CHUẨN JSON NGHIÊM NGẶT gồm 1 key "intent":
        {{
            "intent": "chat" hoặc "diagnose"
        }}
        - Chọn "chat": Nếu người dùng chỉ chào hỏi, cảm ơn, giao tiếp bình thường (VD: "xin chào", "hello", "cảm ơn bác sĩ").
        - Chọn "diagnose": Nếu người dùng kể triệu chứng, hỏi về bệnh, cách chữa (VD: "tôi bị đau đầu", "sốt xuất huyết là gì").
        
        Câu hỏi: "{user_query}"
        """
        try:
            response = self.llm.invoke(system_prompt)
            cleaned_text = response.content.replace("```json", "").replace("```", "").strip()
            intent_data = json.loads(cleaned_text)
            return intent_data.get("intent", "diagnose")
        except Exception as e:
            print(f"[*] Cảnh báo Intent Router: {e}. Mặc định chuyển sang luồng y tế.")
            return "diagnose"

    # -----------------------------------------------------------------------------------
    # LUỒNG 2: XỬ LÝ DỮ LIỆU & LƯU VÀO NEO4J (INGESTION)
    # -----------------------------------------------------------------------------------
    def ingest_medical_data(self, texts: List[str]):
        print("[*] Đang cắt nhỏ văn bản (Chunking) và nạp vào Neo4j...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,     
            chunk_overlap=50,   
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        documents = [Document(page_content=text) for text in texts]
        chunks = text_splitter.split_documents(documents)
        
        print(f"[*] Đã tạo ra {len(chunks)} chunks. Quá trình nhúng bằng PhoBERT bắt đầu...")
        
        self.vector_store = Neo4jVector.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="medical_phobert_index",
            node_label="MedicalChunk",
            text_node_property="text",
            embedding_node_property="embedding"
        )
        
        # Tăng k=15 để lấy đủ ngữ cảnh từ nhiều bệnh khác nhau
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        
        self._build_rag_chain()
        print("[*] Cơ sở dữ liệu và GraphRAG Pipeline đã sẵn sàng!")

    # -----------------------------------------------------------------------------------
    # LUỒNG 3: XÂY DỰNG RAG CHAIN (LANGCHAIN LCEL)
    # -----------------------------------------------------------------------------------
    def _build_rag_chain(self):
        # Template tối ưu: Tự nhiên, ngắn gọn, vào thẳng vấn đề nhưng vẫn đảm bảo Top 5
        template = """Bạn là bác sĩ tư vấn y khoa. 
        Nhiệm vụ của bạn là trả lời câu hỏi liên quan đến [NGỮ CẢNH] đã được cung cấp như tên bệnh, triệu chứng, chẩn đoán, cách phòng ngừa, chuyên khoa, một cách ngắn gọn và chính xác, đồng thời luôn đưa ra lời khuyên y tế phù hợp.
        QUY TẮC TỐI THƯỢNG:
        1. CHỈ dùng thông tin từ [NGỮ CẢNH]. Không đoán bừa hay dùng kiến thức ngoài.
        2. Nếu [NGỮ CẢNH] không có bệnh nào khớp triệu chứng, chỉ cần nói rằng là bạn không biết.
        3. Tuyệt đối không bịa ra thông tin nếu không có ngữ cảnh liên quan được cung cấp.
        [NGỮ CẢNH]:
        {context}

        [CÂU HỎI]: {question}
        Trả lời:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # -----------------------------------------------------------------------------------
    # LUỒNG 4: HÀM GIAO TIẾP TỔNG (ĐIỀU PHỐI)
    # -----------------------------------------------------------------------------------
    def chat(self, user_query: str) -> str:
        if not self.rag_chain:
            return "Vui lòng nạp dữ liệu (ingest_medical_data) trước khi chat."
            
        # Phân loại Intent trước khi chạy RAG
        intent = self.analyze_user_query(user_query)
        
        if intent == "chat":
            return "Xin chào! Tôi là Trợ lý Y khoa. Tôi có thể giúp bạn tra cứu thông tin bệnh lý, phân tích triệu chứng dựa trên cơ sở dữ liệu. Bạn đang cảm thấy như thế nào?"
            
        elif intent == "diagnose":
            try:
                # Đưa vào quy trình RAG nghiêm ngặt
                return self.rag_chain.invoke(user_query)
            except Exception as e:
                return f"Lỗi trong quá trình truy xuất y khoa: {str(e)}"


    def connect_existing_db(self):
        from langchain_community.vectorstores import Neo4jVector
        print("[*] Đang kết nối với Neo4j Database có sẵn...")
        self.vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="medical_phobert_index",
            node_label="MedicalChunk",          # Bổ sung đúng label
            text_node_property="text",          # Bổ sung đúng text property
            embedding_node_property="embedding" # Bổ sung đúng embedding property
        )
        # Khởi tạo retriever với k=15 giống hệt lúc ingest
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})
        # Build lại chain
        self._build_rag_chain()
        print("[*] Đã kết nối Neo4j thành công!")

# CHẠY LOCAL THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    # Tập dữ liệu mẫu đã được mở rộng để test tính năng Top 5
    medical_documents = [
        "Sốt xuất huyết Dengue: Bệnh lây qua muỗi vằn. Triệu chứng gồm sốt cao liên tục 39-40 độ, đau đầu dữ dội, đau hốc mắt, đau cơ khớp, phát ban. Cần theo dõi lượng tiểu cầu trong máu.",
        "Viêm dạ dày cấp tính: Bệnh nhân thường đau vùng thượng vị, buồn nôn, ợ chua, chướng bụng. Nguyên nhân có thể do ngộ độc thực phẩm, nhiễm vi khuẩn HP hoặc lạm dụng thuốc giảm đau NSAID.",
        "Tiểu đường tuýp 2: Triệu chứng biểu hiện qua quy tắc 4 nhiều: khát nhiều, đi tiểu thường xuyên, ăn nhiều nhưng lại sụt cân nhanh chóng. Kèm theo đó là mờ mắt và mệt mỏi.",
        "Cảm cúm thông thường: Sốt nhẹ, ho khan, nghẹt mũi, sổ mũi, mệt mỏi cơ thể. Bệnh thường tự khỏi sau 5-7 ngày nghỉ ngơi.",
        "Ngộ độc thực phẩm: Buồn nôn, nôn mửa liên tục, tiêu chảy, đau quặn bụng, đôi khi kèm theo sốt nhẹ do ăn phải thức ăn ôi thiu.",
        "Sốt rét: Lây qua muỗi Anopheles. Bệnh nhân có những cơn rét run dữ dội, sau đó sốt cao vã mồ hôi. Nhức đầu và mệt mỏi nhiều."
    ]
    
    app = MedicalGraphRAG()
    
    # LƯU Ý: Hàm này tạo các node mới trong Neo4j. 
    # Trong thực tế, chỉ cần gọi 1 lần khi có dữ liệu mới.
    app.ingest_medical_data(medical_documents)
    
    print("\n" + "="*70)
    print("--- TRỢ LÝ Y KHOA GRAPHRAG (PHOBERT + NEO4J + GEMINI) ĐÃ SẴN SÀNG ---")
    print("="*70 + "\n")
    
    while True:
        try:
            query = input("Bạn: ")
            if query.lower() in ['quit', 'exit', 'thoát']:
                print("Tạm biệt! Chúc bạn nhiều sức khỏe.")
                break
            if not query.strip(): 
                continue
            
            answer = app.chat(query)
            print(f"AI:\n{answer}\n")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break