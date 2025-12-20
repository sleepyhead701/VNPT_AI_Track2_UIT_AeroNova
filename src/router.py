import re

class QuestionRouter:
    def classify(self, question_text):
        """
        Trả về: 'READING', 'MATH', hoặc 'KNOWLEDGE'
        (Đã loại bỏ SAFETY vì solver làm tốt hơn)
        """
        text_lower = question_text.lower()

        # 1. Phát hiện câu hỏi Đọc hiểu (Có đoạn văn dài)
        if "đoạn thông tin" in text_lower or "title:" in text_lower or "content:" in text_lower:
            return "READING"
        # Đếm số từ, nếu dài > 150 từ -> Đọc hiểu
        if len(text_lower.split()) > 150:
            return "READING"

        # 2. Phát hiện câu hỏi Toán học/Logic (Cần suy luận CoT)
        math_keywords = [
            r"\$", "giá trị của", "tính", "phương trình", "hàm số", 
            "xác suất", "tọa độ", "tam giác", "hình trụ", "nguyên hàm", 
            "biểu thức", "tích phân", "vector", "tốc độ", "gia tốc",
            "bao nhiêu cách", "tỉ lệ", "lãi suất"
        ]
        for kw in math_keywords:
            if re.search(kw, text_lower):
                return "MATH"

        # 3. Mặc định là Kiến thức chung (Cần RAG hoặc Knowledge nội tại)
        return "KNOWLEDGE"