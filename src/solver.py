
import os
import chromadb
import re
import json
from datetime import date
from typing import List, Dict
from src.api_client import VNPTClient 
class Solver:
    def __init__(self, client: VNPTClient):
        self.client = client
        self.collection_map = {}
        self.local_index = {}

        DB_ROOT = "Vector DB"  # Thư mục chứa 20 folder DB

        if not os.path.exists(DB_ROOT):
            print("❌ Không tìm thấy thư mục 'Vector DB'.")
            self.no_rag = True
            return

        self.no_rag = False
        loaded_count = 0
        for folder_name in os.listdir(DB_ROOT):
            folder_path = os.path.join(DB_ROOT, folder_name)
            if not os.path.isdir(folder_path):
                continue

            try:
                chroma_client = chromadb.PersistentClient(path=folder_path)
                collections = chroma_client.list_collections()
                if collections:
                    collection_name = collections[0].name
                    collection = chroma_client.get_collection(collection_name)
                    self.collection_map[folder_name] = collection
                    existing_data = collection.get() # Lấy tất cả documents
                    if existing_data and 'documents' in existing_data:
                        self.local_index[folder_name] = existing_data['documents']
                    
                    loaded_count += 1
                    print(f"✅ Loaded vector DB: {folder_name} (collection: '{collection_name}')")
                else:
                    print(f"⚠️ Folder '{folder_name}' trống")
            except Exception as e:
                print(f"⚠️ Không load được DB '{folder_name}': {e}")

        print(f"✅ Tổng cộng loaded {loaded_count} vector DBs")
        if loaded_count == 0:
            self.no_rag = True

        
        self.quota_log = "embed_quota.json"
        self.today_quota = self._load_quota()

    def _load_quota(self):
        today = str(date.today())
        if os.path.exists(self.quota_log):
            with open(self.quota_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(today, 0)
        return 0

    def _save_quota(self):
        today = str(date.today())
        data = {}
        if os.path.exists(self.quota_log):
            with open(self.quota_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data[today] = self.today_quota
        with open(self.quota_log, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    domain_keywords = {
        'Toan': ['toán', 'phương trình', 'hàm số', 'đạo hàm', 'tích phân', 'hình học', 'xác suất', 'tam giác', 'nghiệm'],
        'VATLYTHIENVAN': ['vật lý', 'thiên văn', 'vận tốc', 'lực', 'năng lượng', 'động năng', 'hành tinh', 'sao'],
        'Hóa học': ['hóa học', 'phản ứng', 'hợp chất', 'nguyên tử', 'axit', 'bazơ', 'mol', ' dung dịch'],
        'Sinh Học': ['sinh học', 'tế bào', 'gen', 'dna', 'rna', 'protein', 'di truyền', 'loài'],
        'Lịch sử Việt Nam': ['lịch sử', 'hồ chí minh', 'chiến tranh', 'cách mạng', 'khởi nghĩa', 'triều đại', 'năm'],
        'Lịch sử thế giới': ['thế giới', 'thế chiến', 'đế quốc', 'văn minh'],
        'Địa Lý': ['địa lý', 'sông', 'núi', 'biển', 'khí hậu', 'vùng', 'tỉnh', 'thành phố'],
        'Văn học': ['văn học', 'tác phẩm', 'tác giả', 'thơ', 'truyện', 'nhân vật'],
        'Chính trị': ['chính trị', 'đảng', 'chính phủ', 'quốc hội', 'bộ chính trị', 'nghị quyết'],
        'KINH TE': ['kinh tế', 'gdp', 'lãi suất', 'lạm phát', 'doanh nghiệp', 'thị trường'],
        'Luật pháp': ['luật', 'bộ luật', 'quy định', 'nghị định', 'thông tư', 'pháp luật', 'điều'],
        'Y te': ['y tế', 'bệnh', 'thuốc', 'sức khỏe', 'virus', 'bác sĩ'],
        'Tin học': ['tin học', 'máy tính', 'lập trình', 'internet', 'phần mềm', 'dữ liệu', 'mạng'],
        'Thể thao': ['thể thao', 'bóng đá', 'olympic', 'giải đấu', 'huy chương'],
        'Âm nhạc_điện ảnh': ['âm nhạc', 'điện ảnh', 'ca sĩ', 'phim', 'bài hát', 'nhạc sĩ'],
        'Ẩm thực': ['ẩm thực', 'món', 'nấu', 'bánh', 'đặc sản'],
        'Mỹ thuật - Kiến trúc': ['mỹ thuật', 'kiến trúc', 'hội họa', 'đền', 'chùa', 'tượng'],
        'Nghi_dinh': ['nghị định', 'quyết định', 'thông tư', 'xử phạt'],
        'Tôn giáo - Lễ hội': ['tôn giáo', 'lễ hội', 'tết', 'phật', 'chúa', 'thờ'],
        'VNPT_knowledge': ['vnpt', 'vinaphone', 'viễn thông', 'gói cước', 'internet', 'fiber', '4g', '5g']
    }

    def detect_relevant_domains(self, question: str, top_k: int = 5) -> List[str]:
        if self.no_rag: return []
        q_lower = question.lower()
        scores = {}
        for domain, keywords in self.domain_keywords.items():
            # Đếm số từ khóa xuất hiện
            score = sum(2 for kw in keywords if kw in q_lower) # Trọng số 2 cho khớp chính xác
            scores[domain] = score
        
        # Lấy top domains có điểm > 0
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domains = [d for d, s in sorted_domains if s > 0][:top_k]
        
        # Nếu không tìm thấy domain nào, trả về các domain phổ biến
        if not top_domains:
            return [] 
        return top_domains
    def search_local_index(self, question: str, domains: List[str], top_k=3):
        """
        Tìm kiếm dựa trên từ khóa trong RAM. Không tốn API.
        """
        results = []
        q_tokens = set(re.findall(r"\w+", question.lower()))
        # Loại bỏ stopwords cơ bản để tăng độ chính xác
        stopwords = {'là', 'của', 'những', 'cái', 'trong', 'các', 'với', 'và', 'có', 'gì', 'nào', 'tại', 'đâu'}
        q_tokens = q_tokens - stopwords
        
        if not q_tokens: return []

        for domain in domains:
            docs = self.local_index.get(domain, [])
            for doc in docs:
                doc_lower = doc.lower()
                # Tính điểm overlap đơn giản
                score = 0
                for token in q_tokens:
                    if token in doc_lower:
                        score += 1
                
                # Bonus điểm nếu tìm thấy cụm từ dài (Exact phrase match)
                # Ví dụ: "gói cước vd89"
                if len(question) > 10 and question.lower() in doc_lower:
                    score += 10
                
                if score > 0:
                    results.append((score, domain, doc))
        
        # Sắp xếp theo điểm cao nhất
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def retrieve_multi_domain_context(self, question: str, max_docs: int = 8, distance_threshold: float = 0.25) -> str:
        if self.no_rag or self.today_quota >= 490:
            return ""

        domains = self.detect_relevant_domains(question)
        final_docs = []

        # BƯỚC 1: LOCAL SEARCH (MIỄN PHÍ)
        local_results = self.search_local_index(question, domains, top_k=5)
        
        # Kiểm tra chất lượng Local Search
        # Giả sử nếu khớp > 2 từ khóa quan trọng là tốt
        is_local_good = False
        if local_results:
            best_score = local_results[0][0]
            if best_score >= 3: # Ngưỡng chấp nhận
                is_local_good = True
                # Lấy top docs từ local
                for score, dom, doc in local_results[:max_docs]:
                    final_docs.append(f"[{dom}] {doc}")

        if not is_local_good and self.today_quota < 490:
            query_vec = self.client.get_embedding(question)
            if query_vec:
                self.today_quota += 1
                self._save_quota()
                
                vec_results = []
                for domain in domains:
                    collection = self.collection_map.get(domain)
                    if not collection: continue
                    try:
                        results = collection.query(query_embeddings=[query_vec], n_results=2)
                        if results['documents']:
                            docs = results['documents'][0]
                            dists = results['distances'][0]
                            for doc, dist in zip(docs, dists):
                                if dist < distance_threshold:
                                    vec_results.append((dist, domain, doc))
                    except: pass
                
                # Sort vector results by distance (càng nhỏ càng tốt)
                vec_results.sort(key=lambda x: x[0])
                for dist, dom, doc in vec_results[:max_docs]:
                    final_docs.append(f"[{dom}] {doc}")

        # Nếu cả 2 đều không có kết quả, trả về rỗng
        if not final_docs: return ""
        
        # Deduplicate và Join
        unique_docs = list(set(final_docs))
        return "\n\n".join(unique_docs[:max_docs])
    def get_valid_labels(self, choices):
        return [chr(65 + i) for i in range(len(choices))]

    def format_choices(self, choices):
        return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

    def extract_answer_letter(self, text, labels):
        if not text:
            return None
        text = text.upper()
        for l in labels:
            if l in text:
                return l
        m = re.search(r"\b([A-J])\b", text)
        return m.group(1) if m else None

    
    def solve_batch(self, batch_questions, model_type="small", q_type="KNOWLEDGE"):
        if len(batch_questions) == 1:
            q = batch_questions[0]
            if q_type == "MATH":
                return {q['qid']: self.solve_math(q['question'], q['choices'])}
            return {q['qid']: self.solve_knowledge(q['question'], q['choices'])}

        
        prompt = f"""Trả lời danh sách các câu hỏi trắc nghiệm.

⚡ HƯỚNG DẪN:
- Mỗi câu hỏi một ID duy nhất
- Trả lời CHỈ theo format: QID_XXX: A (hoặc B, C, D)
- Mỗi đáp án trên 1 dòng
- Không giải thích, không thêm text khác

"""
        
        if q_type == "MATH":
            prompt += "⚠️ LƯU Ý: Tính toán kỹ lưỡng. Kiểm tra đơn vị, dấu, làm tròn hợp lý.\n\n"
        elif q_type == "READING":
            prompt += "⚠️ LƯU Ý: Dựa trực tiếp vào thông tin trong đề. Không sáng tạo.\n\n"
        else:
            prompt += "⚠️ LƯU Ý: Sử dụng kiến thức chứng thực.\n\n"
        
        prompt += "DANH SÁCH CÂU HỎI:\n\n"
        
        for q in batch_questions:
            context = self.retrieve_multi_domain_context(q['question'])
            prompt += f"QID: {q['qid']}\n"
            prompt += f"Câu hỏi: {q['question']}\n"
            prompt += f"Lựa chọn:\n{self.format_choices(q['choices'])}\n"
            if context:
                prompt += f"Thông tin: {context[:300]}\n"
            prompt += "\n"
        
        prompt += "KẾT QUẢ (format: QID_XXX: A):\n"

        messages = [{"role": "user", "content": prompt}]
        res = self.client.call_chat(model_type, messages, temperature=0.12)

        results = {}
        if res:
            
            matches = re.findall(r"QID[_:]?\s*(\d+|[^\s:]+)\s*[:=]\s*([A-J])", res, re.IGNORECASE)
            for qid, ans in matches:
                
                for q in batch_questions:
                    if str(q['qid']).endswith(str(qid)):
                        results[q['qid']] = ans.upper()
                        break
            
            
            if not results:
                for q in batch_questions:
                    pattern = rf"{re.escape(str(q['qid']))}\s*[:=]\s*([A-J])"
                    m = re.search(pattern, res, re.IGNORECASE)
                    if m:
                        results[q['qid']] = m.group(1).upper()

        
        if len(results) < len(batch_questions) * 0.7:
            print(f"⚠️ Batch failed ({len(results)}/{len(batch_questions)}). Switching to sequential fallback ({q_type})")
            for q in batch_questions:
                if q_type == "MATH":
                    ans = self.solve_math(q['question'], q['choices'])
                else:
                    ans = self.solve_knowledge(q['question'], q['choices'])
                results[q['qid']] = ans

        return results

    
    def solve_math(self, question, choices):
        labels = self.get_valid_labels(choices)
        last_char = labels[-1]
        context = self.retrieve_multi_domain_context(question)
        prompt = f"""STEM EXPERT: Solve this mathematics/physics/chemistry problem with precision and logic.

📌 PROBLEM: {question}

📋 OPTIONS:
{self.format_choices(choices)}

💡 REFERENCE KNOWLEDGE:
{context if context else "(No reference knowledge provided - use internal knowledge)"}

⚡ SOLUTION METHODOLOGY:
1.ANALYZE: What data is given? What is the target variable? What is the problem type?
2.FORMULATE: Select the appropriate formulas or methods.
3.COMPUTE: Calculate step-by-step. Pay strict attention to units, signs (+/-), and rounding.
4 VERIFY: Is the result logical and reasonable?
5 CHECK: Check the result again using a different perspective or estimation.
6 MATCH: Compare the calculated result with the options. Select the closest one.

⚠️ CRITICAL RULES:
- 📐 Check: Units (m, cm, V, A...), signs, and rounding precision.
- 🔍 If there is a small discrepancy/rounding error, select the closest option.
- ❌ NO guessing. You MUST rely on mathematical logic.
- ✅ Return ONLY one letter from A to {last_char}

🎯 OUTPUT FORMAT:
[Data: ...] -> [Formula: ...] -> [Calculation: ...] -> [Answer: ONLY THE LETTER]"""
        messages = [{"role": "user", "content": prompt}]
        res = self.client.call_chat("large", messages, temperature=0.1)
        ans = self.extract_answer_letter(res, labels)
        return ans or "A"

    
    def solve_knowledge(self, question, choices):
        labels = self.get_valid_labels(choices)
        context = self.retrieve_multi_domain_context(question)
        prompt = f"""TASK: Answer the multiple-choice question using verified knowledge and the provided context.

❓ QUESTION: {question}

📋 OPTIONS:
{self.format_choices(choices)}

💡 REFERENCE CONTEXT:
{context if context else "(No context provided - use internal knowledge)"}

⚡ REASONING PROCESS:
1️⃣ ANALYZE: Identify keywords and specific requirements in the question.
2️⃣ VERIFY WITH CONTEXT: Check if the provided context contains the answer. Treat context as GROUND TRUTH.
3️⃣ ELIMINATE: Rule out options that contradict the context or established facts.
4️⃣ COMPARE: Match the remaining options against the evidence.
5️⃣ DECIDE: Select the most accurate option (or the most probable one if uncertain).

⚠️ CRITICAL RULES:
- ✅ PRIORITIZE CONTEXT: If the answer is in the context, you MUST use it.
- ❌ NO HALLUCINATION: Do not invent information not present in the context or reality.
- ❌ AVOID AMBIGUITY: Choose the definitive answer.
- ✅ RETURN ONLY ONE LETTER: A, B, C, D, or E.

🎯 OUTPUT FORMAT:
[Context Analysis: ...] -> [Elimination: ...] -> [Answer: ONLY THE LETTER]"""
        messages = [{"role": "user", "content": prompt}]
        res = self.client.call_chat("small", messages, temperature=0.10)
        ans = self.extract_answer_letter(res, labels)
        return ans or "A"

    
    def solve_reading(self, question, choices):
        labels = self.get_valid_labels(choices)
        context = self.retrieve_multi_domain_context(question, max_docs=3) if len(question.split()) < 150 else ""
        prompt = f"""TASK: Perform precise reading comprehension based strictly on the provided text snippets.

📖 SOURCE TEXT:
{context}

❓ QUESTION:
{question}

📋 OPTIONS:
{self.format_choices(choices)}

⚡ ANALYSIS PROCESS:
1️⃣ LOCATE: Scan the text to find keywords related to the question.
2️⃣ VERIFY: Compare each option against the specific details in the text.
3️⃣ ELIMINATE: Discard options that are not mentioned or contradict the text.
4️⃣ CONCLUDE: Select the option that is most supported by the evidence in the text.

⚠️ CRITICAL RULES:
- 🛑 RESTRICTED SCOPE: Answer ONLY based on the provided "SOURCE TEXT".
- ❌ NO OUTSIDE KNOWLEDGE: Do not use external facts, even if they are true. If it's not in the text, it's not the answer.
- ❌ NO HALLUCINATION: Do not infer details not explicitly stated.
- ✅ RETURN ONLY ONE LETTER: A, B, C, D, or E.

🎯 OUTPUT FORMAT:
[Evidence mapping: ...] -> [Answer: ONLY THE LETTER]"""
        messages = [{"role": "user", "content": prompt}]
        res = self.client.call_chat("large", messages, temperature=0.08)
        ans = self.extract_answer_letter(res, labels)
        return ans or self.solve_knowledge(question, choices)

    
    def solve_safety_local(self, question: str, choices):
        
        # --- DANH SÁCH TỪ KHÓA ---
        refusal_candidates = [
            "tôi không thể", "không thể trả lời", "từ chối", "không thể chia sẻ",
            "vi phạm", "bất hợp pháp", "không hỗ trợ", "nằm ngoài phạm vi",
            "không được phép", "cung cấp thông tin về cách", 
            "liên quan đến vấn đề này" 
        ]

        bad_topics = [
            # Kinh tế/Pháp luật
            "trốn", "lậu", "giả", "nhái", "tránh", "né", "qua mặt",
            "đình chỉ", "cấm", "tước", "thu hồi",
            "trái quy định", "trái luật", "sai quy định", "sai luật",
            
            # Chính trị/An ninh
            "phản động", "chống phá", "lật đổ", "biểu tình", "bạo loạn",
            "xúc phạm", "bôi nhọ", "nói xấu", "xuyên tạc", "lãnh tụ",
            "bí mật", "quốc gia", "nội bộ",
            
            # Tệ nạn/Bạo lực
            "ma túy", "kích thích", "bom", "mìn", "vũ khí", "súng",
            "giết", "tự tử", "đánh", "chém", "khiêu dâm", "sex",
            "hack", "crack", "ddos", "tấn công mạng"
        ]

        # Nhóm hành động (Câu hỏi):
        bad_actions = [
            "làm thế nào", "cách nào", "phương pháp nào", "làm sao",
            "hướng dẫn", "chỉ tôi", "muốn", "ở đâu", "để"
        ]
        
        q_lower = question.lower()
        c_lower = [str(c).lower() for c in choices]
        labels = self.get_valid_labels(choices)

        # BƯỚC 1: Quét Đáp án (Tìm nghi phạm)
        potential_idx = -1
        for i, choice in enumerate(c_lower):
            for pat in refusal_candidates:
                if pat in choice:
                    potential_idx = i
                    break
            if potential_idx != -1: break
        
        # Nếu KHÔNG thấy đáp án từ chối -> Chắc chắn không phải Safety (hoặc là Safety dạng ẩn, hiếm gặp)
        if potential_idx == -1:
            return None

        has_bad_topic = any(bt in q_lower for bt in bad_topics)
        
        # Trường hợp A: Câu hỏi có topic xấu (VD: bom, ma túy) -> CHỐT LUÔN
        if has_bad_topic:
            return labels[potential_idx]
            
        # Trường hợp B: Câu hỏi không có topic xấu rõ ràng, nhưng có vẻ đang hỏi "Cách làm" (Action)
        # VD: "Làm sao để không bị công an bắt?" (Không có keyword mạnh, nhưng ý xấu)
        has_bad_action = any(ba in q_lower for ba in bad_actions)
        if has_bad_action and ("luật" not in q_lower) and ("quy định" not in q_lower):
            return labels[potential_idx]

        # Trường hợp C: (Ngoại lệ mạnh) Nếu đáp án từ chối cực kỳ rõ ràng kiểu AI
        # VD: "Tôi là trí tuệ nhân tạo, tôi không thể..." -> Chốt luôn không cần check câu hỏi
        refusal_content = c_lower[potential_idx]
        if "tôi là" in refusal_content or "trí tuệ nhân tạo" in refusal_content or "chính sách an toàn" in refusal_content:
            return labels[potential_idx]

        # Nếu chỉ có đáp án từ chối mà câu hỏi hoàn toàn trong sáng (VD: hỏi về Luật)
        # Thì khả năng đó là đáp án Sai. Trả về None để các Solver khác giải quyết.
        return None
    
    
    def solve_safety(self, question, choices):
        """Xử lý Safety: thử regex trước, nếu fail thì gọi LLM"""
        
        ans = self.solve_safety_local(question, choices)
        if ans:
            return ans
        
        
        labels = self.get_valid_labels(choices)
        prompt = f"""Kiểm tra xem câu hỏi này có nội dung nhạy cảm hoặc yêu cầu nội dung vi phạm pháp luật hay không:

❓ CÂU HỎI: {question}

📋 LỰA CHỌN:
{self.format_choices(choices)}

⚡ HƯỚNG DẪN:
- Nếu câu hỏi yêu cầu nội dung KHÔNG HỢP PHÁP hoặc NHẠY CẢM:
  → Chọn đáp án từ chối/không thể trả lời (chứa cụm từ "tôi không thể")
- Nếu câu hỏi bình thường:
  → Trả lời như câu hỏi thường

⚠️ QUAN TRỌNG: Chỉ trả lời 1 chữ cái (A, B, C hoặc D)

🎯 Đáp án: """
        
        messages = [{"role": "user", "content": prompt}]
        res = self.client.call_chat("small", messages, temperature=0.05)
        ans = self.extract_answer_letter(res, labels)
        
        return ans or "A"