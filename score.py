import pandas as pd
import os
import re

#================= CẤU HÌNH ĐƯỜNG DẪN =================
VAL_FILE = "Testing/ANS_DA_DANH_SO.txt"  # File đáp án gốc (.txt)
PRED_FILE = "output/submission_test15.csv"  # File dự đoán (.csv)
# ========================================================

# VAL_FILE = "Testing/ANS_VAL_SAFETY.txt"  # File đáp án gốc (.txt)
# PRED_FILE = "output/safety2.csv"  # File dự đoán (.csv)
def load_ground_truth(file_path):
    """Đọc file txt đáp án gốc (Format: ID Answer)"""
    ground_truth = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Tách bằng khoảng trắng hoặc Tab
                parts = re.split(r'\s+', line)
                if len(parts) >= 2:
                    qid = parts[0].strip()
                    ans = parts[1].strip().upper()
                    
                    # Chuẩn hóa ID nếu cần (ví dụ đảm bảo test_xxxx)
                    # Ở đây giả định file txt đã có sẵn prefix test_
                    ground_truth[qid] = ans
                        
        print(f"✅ Đã đọc {len(ground_truth)} đáp án gốc từ {file_path}")
        return ground_truth
    except Exception as e:
        print(f"❌ Lỗi đọc file đáp án gốc: {e}")
        return {}

def load_predictions(file_path):
    """Đọc file csv dự đoán"""
    predictions = {}
    try:
        df = pd.read_csv(file_path)
        if 'qid' not in df.columns or 'answer' not in df.columns:
            print("⚠️ File CSV thiếu cột 'qid' hoặc 'answer'")
            return {}
            
        for _, row in df.iterrows():
            qid = str(row['qid']).strip()
            ans = str(row['answer']).strip().upper()
            predictions[qid] = ans
            
        print(f"✅ Đã đọc {len(predictions)} câu trả lời từ {file_path}")
        return predictions
    except Exception as e:
        print(f"❌ Lỗi đọc file dự đoán: {e}")
        return {}

def calculate_score():
    if not os.path.exists(VAL_FILE):
        print(f"❌ Không tìm thấy file gốc: {VAL_FILE}")
        return
    if not os.path.exists(PRED_FILE):
        print(f"❌ Không tìm thấy file dự đoán: {PRED_FILE}")
        return

    # 1. Load dữ liệu
    ground_truth = load_ground_truth(VAL_FILE)
    predictions = load_predictions(PRED_FILE)

    if not ground_truth:
        return

    # 2. So sánh
    correct = 0
    total_checked = 0
    missing = 0
    wrong_list = []

    # Duyệt qua tất cả câu trong đáp án gốc
    sorted_qids = sorted(ground_truth.keys())

    for qid in sorted_qids:
        true_ans = ground_truth[qid]
        pred_ans = predictions.get(qid, "N/A")
        
        total_checked += 1
        
        if pred_ans == "N/A":
            missing += 1
        elif pred_ans == true_ans:
            correct += 1
        else:
            wrong_list.append((qid, true_ans, pred_ans))

    # 3. Kết quả
    accuracy = (correct / total_checked) * 100 if total_checked > 0 else 0

    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ CHẤM ĐIỂM")
    print("=" * 50)
    print(f"Tổng số câu trong đáp án gốc: {total_checked}")
    print(f"Số câu làm ĐÚNG:              {correct}")
    print(f"Số câu làm SAI:               {len(wrong_list)}")
    print(f"Số câu THIẾU (chưa làm):      {missing}")
    print("-" * 50)
    print(f"🏆 ĐỘ CHÍNH XÁC (ACCURACY):   {accuracy:.2f}%")
    print("=" * 50)

    # In danh sách câu sai (Để debug)
    if wrong_list:
        print(f"\n📝 DANH SÁCH CÁC CÂU SAI ({len(wrong_list)} câu):")
        print(f"{'QID':<12} | {'Gốc':<5} | {'Chọn':<5}")
        print("-" * 30)
        for w in wrong_list[:20]:  # In 20 câu đầu thôi cho gọn
            print(f"{w[0]:<12} | {w[1]:<5} | {w[2]:<5}")
        if len(wrong_list) > 20:
            print(f"... và {len(wrong_list)-20} câu sai khác.")

if __name__ == "__main__":
    calculate_score()