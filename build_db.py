__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import zipfile
import shutil
import chromadb
import re
import unicodedata
import time
from tqdm import tqdm
from src.api_client import VNPTClient
from chromadb.api.models.Collection import Collection
import subprocess

# ================= CẤU HÌNH =================
ZIP_FILE = "knowledge_data.zip"
EXTRACT_DIR = "knowledge_data"    
DB_ROOT_DIR = "Vector DB" 

# ================= HÀM XỬ LÝ TEXT (CHUẨN CŨ) =================

def sanitize_filenames(root_dir):
    """
    Quét toàn bộ thư mục và đổi tên file chứa ký tự đặc biệt.
    Ví dụ: 'Am_nhac_&_Dien_anh.txt' -> 'Am_nhac_va_Dien_anh.txt'
    """
    print(f"-> Scanning for special characters in filenames in '{root_dir}'...")
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if "&" in filename:
                old_path = os.path.join(dirpath, filename)
                
                # Thay thế & bằng _va_ cho dễ đọc và an toàn
                new_filename = filename.replace("&", "_va_")
                
                # Hoặc muốn đơn giản hơn thì thay bằng gạch dưới:
                # new_filename = filename.replace("&", "_")
                
                new_path = os.path.join(dirpath, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    count += 1
                    # print(f"   [Renamed] {filename} -> {new_filename}")
                except Exception as e:
                    print(f"   ⚠️ Could not rename {filename}: {e}")
                    
    if count > 0:
        print(f"   ✅ Renamed {count} files containing '&'.")
    else:
        print("   ✅ No files needed renaming.")

def slugify(text: str):
    """
    Chuyển tên file thành tên Collection hợp lệ.
    QUAN TRỌNG: Lấy phần ĐUÔI của tên file để tránh trùng lặp.
    """
    if text.lower().endswith('.txt'):
        text = text[:-4]
    
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', '_', text) 
    text = re.sub(r'_+', '_', text).strip('_')
    
    if len(text) > 60:
        text = text[-60:]
        text = text.strip('_')
        # Nếu cắt xong mà ký tự đầu là số hoặc _ thì thêm tiền tố
        if not text[0].isalpha():
            text = "db_" + text
            
    if len(text) < 3: text = text + "_col"
    return text

def extract_chunks_from_file_format(text):
    """
    Hàm cắt chunk CHUẨN: Dựa vào dòng 'CHUNK x/y' và đường kẻ '===='
    """
    chunks = []
    # 1. Tách văn bản dựa trên dòng kẻ ngang dài (===...)
    parts = re.split(r'={10,}', text)
    
    # 2. Duyệt qua các phần
    # Logic: Nếu phần i là header "CHUNK...", thì phần i+1 là nội dung
    for i in range(len(parts) - 1):
        current_part = parts[i].strip()
        next_part = parts[i+1].strip()
        
        # Regex tìm chữ "CHUNK" theo sau là số
        if re.search(r'CHUNK\s+\d+', current_part):
            if next_part: 
                chunks.append(next_part)
                
    return chunks

# ================= HÀM XỬ LÝ DB =================

def embed_and_save_batch(collection: Collection, filename: str, chunks: list, client: VNPTClient):
    """Embed và lưu vào collection theo batch"""
    ids = []
    embeddings = []
    docs = []
    metadatas = []
    
    # Chuẩn bị dữ liệu
    for idx, chunk in enumerate(chunks):
        c_id = f"chunk_{idx+1}" # ID chuẩn: chunk_1, chunk_2...
        ids.append(c_id)
        docs.append(chunk)
        # Metadata lưu tên file gốc để tra cứu ngược
        metadatas.append({"source_file": filename, "chunk_index": c_id})

    final_ids = []
    final_embeds = []
    final_docs = []
    final_metas = []

    for i, doc in enumerate(docs):
        try:
            emb = client.get_embedding(doc)
            # Kiểm tra vector rỗng hoặc lỗi
            if emb and len(emb) > 0 and sum(emb) != 0:
                final_ids.append(ids[i])
                final_embeds.append(emb)
                final_docs.append(doc)
                final_metas.append(metadatas[i])
        except:
            pass

    # Ghi vào DB (Batch write)
    if final_ids:
        batch_size = 50
        for i in range(0, len(final_ids), batch_size):
            collection.add(
                ids=final_ids[i:i+batch_size],
                embeddings=final_embeds[i:i+batch_size],
                documents=final_docs[i:i+batch_size],
                metadatas=final_metas[i:i+batch_size]
            )
            
    return len(final_ids)
def robust_unzip(zip_path, extract_to):
    print(f"-> Unzipping {zip_path}...")
    
    # Cách 1: Thử dùng Python zipfile (Chuẩn)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("   [OK] Unzip successful with Python zipfile.")
        return True
    except Exception as e:
        print(f"   ⚠️ Python zipfile failed: {e}")
        print("   -> Attempting system unzip (Linux command)...")

    # Cách 2: Thử dùng lệnh hệ thống 'unzip' (Dự phòng cho Linux/Docker)
    try:
        # Lệnh: unzip -o -q file.zip -d folder
        # -o: overwrite, -q: quiet
        result = subprocess.run(["unzip", "-o", "-q", zip_path, "-d", "."], capture_output=True, text=True)
        if result.returncode == 0:
            print("   [OK] System unzip successful.")
            return True
        else:
            print(f"   ❌ System unzip failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ System command failed: {e}")
        
    return False

def setup_environment():
    print("=== SETUP DATA PIPELINE (EXACT CHUNK VERSION) ===")
    
    # 1. UNZIP
    if not os.path.exists(EXTRACT_DIR):
        if os.path.exists(ZIP_FILE):
            print(f"-> Unzipping {ZIP_FILE}...")
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(".")
        else:
            print(f"⚠️ Không thấy file zip. Kiểm tra folder {EXTRACT_DIR}...")
    
    if not os.path.exists(EXTRACT_DIR):
        print("❌ Lỗi: Không có dữ liệu đầu vào.")
        return

    # 2. BUILD DB THEO CẤU TRÚC FOLDER
    client = VNPTClient()
    
    # Lấy danh sách các Topic (Folder con trong knowledge_data)
    # Ví dụ: Toan, Y te, VNPT_knowledge...
    topics = [d for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d))]
    
    for topic in tqdm(topics, desc="Processing Topics"):
        # Đường dẫn: knowledge_data/Toan
        topic_src_path = os.path.join(EXTRACT_DIR, topic)
        
        # Đường dẫn DB: chroma_db_per_file/Toan
        # Đây chính là yêu cầu của bạn: Mỗi chủ đề là 1 folder DB riêng
        topic_db_path = os.path.join(DB_ROOT_DIR, topic)
        
        # Nếu DB chủ đề này đã có dữ liệu thì bỏ qua (Resume)
        if os.path.exists(topic_db_path) and len(os.listdir(topic_db_path)) > 0:
            continue
            
        os.makedirs(topic_db_path, exist_ok=True)
        
        # Khởi tạo Client riêng cho Topic này
        try:
            chroma_client = chromadb.PersistentClient(path=topic_db_path)
        except:
            continue

        # Duyệt các file txt trong Topic
        txt_files = [f for f in os.listdir(topic_src_path) if f.endswith(".txt")]
        
        for txt_file in txt_files:
            file_path = os.path.join(topic_src_path, txt_file)
            
            # Tên collection = Tên file (đã xử lý đuôi)
            col_name = slugify(txt_file)
            
            try:
                # Đọc file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Cắt Chunk (Logic chuẩn)
                chunks = extract_chunks_from_file_format(content)
                
                if not chunks: continue # Bỏ qua nếu file rỗng/sai format

                # Tạo Collection
                # Dùng get_or_create để tránh lỗi nếu chạy lại
                collection = chroma_client.get_or_create_collection(name=col_name)
                
                # Nếu collection chưa có data thì mới embed
                if collection.count() == 0:
                    count = embed_and_save_batch(collection, txt_file, chunks, client)
                    # print(f"   -> {txt_file}: {count} chunks")
                    
            except Exception as e:
                print(f"Error {txt_file}: {e}")

    print("\n=== DATA SETUP DONE ===")

if __name__ == "__main__":
    setup_environment()