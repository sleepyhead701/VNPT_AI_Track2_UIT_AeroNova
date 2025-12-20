VNPT AI Hackathon 2025 - Track 2 Solution
📚 Tổng quan Giải pháp (Executive Summary)
Repository này chứa mã nguồn giải pháp cho bài toán Hỏi đáp đa lĩnh vực (Multi-domain Q&A). Giải pháp của chúng tôi tập trung giải quyết bài toán cốt lõi: "Làm sao để đạt độ chính xác cao nhất trên 2000 câu hỏi test với giới hạn tài nguyên cực kỳ ngặt nghèo (500 API requests/ngày)?".
Phương pháp tiếp cận của chúng tôi là xây dựng một Hệ thống RAG Phân tầng (Hierarchical RAG) kết hợp với Cơ chế Định tuyến Động (Dynamic Routing), ưu tiên độ trễ thấp và an toàn thông tin tuyệt đối.
💡 1. Kiến trúc Hệ thống & Luồng xử lý (Architecture & Pipeline)
Hệ thống được thiết kế theo kiến trúc Pipeline Tuần tự có Điều kiện (Conditional Sequential Pipeline), giúp tối ưu hóa tài nguyên cho từng loại câu hỏi.
Sơ đồ Luồng xử lý (Processing Flow)
code
Mermaid
graph TD
    A[Input Question] --> B{Layer 1: Safety Filter};
    B -- Unsafe Detected --> C[Refusal Response < 0.01s];
    B -- Safe --> D{Layer 2: Router & Classifier};
    
    D -- Math/Logic Keywords --> E[Strategy: CoT + Large Model];
    D -- Reading/Context Keywords --> F[Strategy: Deep RAG + Large Model];
    D -- General Knowledge --> G[Strategy: Hybrid RAG + Small Model];
    
    G --> H{Layer 3: Hybrid Retrieval};
    H -- Local Index Hit (High Score) --> I[RAM Context (O(1) Latency)];
    I --> K[Answer Generation];
    H -- Local Index Miss --> J{Quota Available?};
    J -- Yes --> L[Vector Search (API Embedding)];
    J -- No --> M[Internal Knowledge (Fallback)];
    
    L --> K;
    M --> K;
    
    C --> Z[Final Output];
    E --> Z;
    F --> Z;
    K --> Z;
Chi tiết Logic Suy luận (Reasoning Logic):
Lớp bảo vệ (Safety Layer - Zero Latency):
Logic: Thay vì dùng Model để check safety (tốn phí, chậm), chúng tôi sử dụng thuật toán Keyword Matching & Regex hai chiều (kiểm tra cả câu hỏi và các đáp án "từ chối" tiềm năng).
Hiệu quả: Phát hiện các câu hỏi vi phạm (chính trị, bạo lực, trốn thuế...) trong < 0.005s, tiết kiệm 100% API quota cho các câu này.
Lớp định tuyến (Routing Layer):
Phân loại câu hỏi dựa trên đặc trưng ngôn ngữ:
Toán học/Logic: Cần khả năng suy luận -> Dùng Large Model với Prompt Chain-of-Thought (CoT).
Đọc hiểu: Cần ngữ cảnh dài -> Dùng Large Model với ngữ cảnh trích xuất.
Kiến thức chung: Cần tra cứu nhanh -> Dùng Small Model với Hybrid RAG.
Lớp truy xuất thông tin (Retrieval Layer - The Innovation):
Đây là điểm sáng tạo nhất để vượt qua giới hạn 500 requests. Chúng tôi sử dụng chiến lược Hybrid Search:
Bước 1 (RAM Search): Load toàn bộ tri thức vào RAM dưới dạng Inverted Index. Tìm kiếm từ khóa (BM25-like). Nếu độ khớp cao -> Dùng luôn (Miễn phí, Tức thì).
Bước 2 (Vector Search): Chỉ khi Bước 1 thất bại và còn Quota, hệ thống mới gọi API Embedding để tìm kiếm ngữ nghĩa.
🚀 2. Chiến lược Tối ưu & Sáng tạo (Optimization Highlights)
Đây là phần chúng tôi tối ưu hóa để đạt điểm tuyệt đối cho tiêu chí "Ý tưởng & cách tiếp cận":
2.1. Chiến thuật "One-File-One-Collection" (Granularity Optimization)
Vấn đề: Các phương pháp RAG truyền thống thường gộp chung dữ liệu vào một Vector Store lớn, dẫn đến việc truy xuất bị nhiễu (Noise) khi các chủ đề khác nhau có từ khóa giống nhau.
Giải pháp: Chúng tôi chia nhỏ dữ liệu: 1 File văn bản gốc = 1 ChromaDB Collection.
Kết quả: Khi câu hỏi đề cập đến một văn bản cụ thể (ví dụ "Nghị định 100"), hệ thống định vị chính xác Collection tương ứng, loại bỏ hoàn toàn nhiễu từ các văn bản khác, tăng độ chính xác của RAG lên đáng kể.
2.2. Kỹ thuật "Double-Check" cho bài toán STEM
Vấn đề: LLM thường hay ảo giác hoặc tính toán sai số học.
Giải pháp: Sử dụng Prompt Engineering nâng cao (Tiếng Anh) yêu cầu mô hình thực hiện giải bài toán theo 2 cách khác nhau trong cùng một lần suy luận (Self-Consistency) và so sánh kết quả trước khi đưa ra đáp án cuối cùng.
2.3. Cơ chế "Robust Batching" (Latency Optimization)
Vấn đề: Xử lý tuần tự (Sequential) quá chậm, nhưng xử lý song song (Batch) dễ bị lỗi nếu một mẫu dữ liệu hỏng.
Giải pháp: Xây dựng cơ chế Smart Fallback. Hệ thống mặc định chạy Batch để tối đa tốc độ. Nếu Batch gặp lỗi, tự động chuyển sang chế độ Sequential cho riêng Batch đó để đảm bảo không bỏ sót câu hỏi nào, cân bằng hoàn hảo giữa Tốc độ và Độ ổn định.
🛠 3. Quy trình Xử lý Dữ liệu (Data Processing)
3.1. Nguồn dữ liệu (Data Sources)
Knowledge Base được xây dựng từ các nguồn chính thống, đảm bảo tính pháp lý và chính xác:
Thư viện Pháp luật & Cổng thông tin Chính phủ: Luật, Nghị định, Thông tư.
Wikipedia Tiếng Việt: Lịch sử, Địa lý, Văn hóa.
Tài liệu STEM: Công thức toán học, vật lý, hóa học.
VNPT Official: Thông tin gói cước, dịch vụ viễn thông.
3.2. Tiền xử lý (Pre-processing)
Semantic Chunking: Tách văn bản dựa trên cấu trúc ngữ nghĩa (định dạng CHUNK x/y trong dữ liệu gốc) thay vì cắt theo số ký tự cố định. Điều này giữ nguyên vẹn ý nghĩa của các điều luật hoặc đoạn văn.
Metadata Tagging: Gán nhãn chủ đề (Topic) cho từng chunk để phục vụ việc lọc (Filtering) khi truy vấn.
⚙️ 4. Hướng dẫn Khởi tạo & Chạy (User Guide)
4.1. Cấu trúc thư mục
code
Text
.
├── Dockerfile
├── requirements.txt
├── knowledge_data.zip      # Dữ liệu gốc (Bắt buộc)
├── predict.py              # Entry-point chính
└── src/
    ├── api_client.py       # Quản lý API & Rate Limit
    ├── config.py           # Cấu hình hệ thống
    ├── router.py           # Phân loại câu hỏi
    ├── setup_data.py       # Module khởi tạo & Indexing
    └── solver.py           # Core Logic (Hybrid RAG, Safety)
4.2. Chạy với Docker
Hệ thống tự động thực hiện: Giải nén dữ liệu -> Build Vector DB (nếu chưa có) -> Inference -> Xuất báo cáo.
code
Bash
# 1. Build Image
sudo docker build -t team_submission .

# 2. Run Container (Mount dữ liệu và thư mục output)
sudo docker run --gpus all \
  -v /absolute/path/to/data:/app/data \
  -v /absolute/path/to/output:/app/output \
  team_submission
Input: /app/data/private_test.json
Output: /app/output/submission.csv và /app/output/submission_time.csv
📚 5. Thư viện sử dụng
pandas: Xử lý dữ liệu bảng.
chromadb: Vector Database (Local).
tqdm: Hiển thị tiến trình.
requests: Gọi API VNPT.
pysqlite3-binary: Đảm bảo tương thích DB trên môi trường Linux.
Cam kết: Giải pháp này được tối ưu hóa hoàn toàn cho luật chơi và giới hạn tài nguyên của VNPT AI Hackathon 2025, đảm bảo tính khả thi, hiệu quả và độ tin cậy cao nhất.
Team Name: [TÊN ĐỘI CỦA BẠN]
Contact: [EMAIL CỦA BẠN]
Team Name: UIT_AeroNova
Contact: 24521374@gm.uit.edu.vn
