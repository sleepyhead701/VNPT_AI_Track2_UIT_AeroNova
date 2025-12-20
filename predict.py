__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
import pandas as pd
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm
from src.api_client import VNPTClient
from src.router import QuestionRouter
from src.solver import Solver
from src.config import INPUT_PATH, OUTPUT_PATH, OUTPUT_TIME_PATH

def save_checkpoint(results):
    """Lưu checkpoint an toàn"""
    try:
        df_out = pd.DataFrame(results)
        df_out.to_csv(OUTPUT_PATH, index=False)
        if 'time' not in pd.DataFrame(results).columns:
             for r in results: r['time'] = r.get('time', 0.0)
             
        df_time = pd.DataFrame(results)[['qid', 'answer', 'time']]
        df_time.to_csv(OUTPUT_TIME_PATH, index=False)
        print(f"   ✅ Checkpoint saved: {len(results)} questions")
    except Exception as e:
        print(f"   ⚠️ Failed to save checkpoint: {e}")

def main():
    print("=== STARTING PIPELINE WITH TIMING MEASUREMENT ===")
    
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input not found at {INPUT_PATH}")
        return

    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
    except:
        try:
            df = pd.read_csv(INPUT_PATH)
            all_questions = []
            for _, row in df.iterrows():
                all_questions.append({
                    "qid": str(row.get('id', row.get('qid'))),
                    "question": row['question'],
                    "choices": [row['option_1'], row['option_2'], row['option_3'], row['option_4']]
                })
        except Exception as e:
            print(f"Error reading data file: {e}")
            return
    
    print(f"Total questions loaded: {len(all_questions)}")

    
    
    
    results = []
    processed_qids = set()

    if os.path.exists(OUTPUT_TIME_PATH):
        try:
            df_done = pd.read_csv(OUTPUT_TIME_PATH)
            if 'qid' in df_done.columns and 'answer' in df_done.columns:
                results = df_done.to_dict('records')
                processed_qids = set(str(r['qid']) for r in results)
                print(f"✅ Resuming from checkpoint: {len(results)} questions done")
        except Exception as e:
            print(f"⚠️ Could not read checkpoint ({e}). Starting fresh.")

    
    
    
    router = QuestionRouter()
    client = VNPTClient()
    solver = Solver(client)
    buckets = {
        "MATH": [],
        "READING": [],
        "KNOWLEDGE": []
    }
    
    questions_to_process = [q for q in all_questions if str(q['qid']) not in processed_qids]
    
    if not questions_to_process:
        print("✅ All questions completed!")
        return
    
    print(f"\nClassifying {len(questions_to_process)} remaining questions...")
    safety_count = 0
    for q in tqdm(questions_to_process, desc="Filtering Safety" ):
        qid = q['qid']
        start_time = time.time()
        ans_safety = solver.solve_safety_local(q['question'], q['choices'])
        end_time = time.time()
        infer_time = end_time - start_time
        # BƯỚC 1: SAFETY CHECK (Ưu tiên số 1 - Local Regex)
        # Nếu phát hiện Safety -> Giải quyết luôn, KHÔNG đưa vào buckets
        ans_safety = solver.solve_safety_local(q['question'], q['choices'])
        if ans_safety:
            results.append({"qid": qid, "answer": ans_safety,"time": infer_time})
            safety_count += 1
            continue

        # BƯỚC 2: ROUTING (Nếu không phải Safety)
        q_type = router.classify(q['question'])
        
        if q_type == "MATH":
            buckets["MATH"].append(q)
        elif q_type == "READING":
            buckets["READING"].append(q)
        else:
            buckets["KNOWLEDGE"].append(q) # Mặc định là Knowledge

    # Lưu checkpoint sau khi lọc Safety
    if safety_count > 0:
        print(f"   🔒 Processed {safety_count} SAFETY questions locally.")
        save_checkpoint(results)

    print(f"\n📊 Remaining Buckets Distribution:")
    for k, v in buckets.items():
        print(f"   {k}: {len(v)} questions")

    
    def run_smart_batch(questions: list, model:str, max_chars, type_label):
        """Xử lý batch với error handling tốt"""
        if not questions:
            return
        
        print(f"\n📦 Processing {len(questions)} {type_label} -> {model.upper()}")
        
        
        questions.sort(key=lambda x: len(x['question']))
        
        current_batch = []
        current_chars = 0
        max_batch_size = 6 if model == "large" else 15
        
        pbar = tqdm(total=len(questions))
        
        for i, q in enumerate(questions):
            q_len = len(q['question'])
            
            
            should_break = (
                current_batch and 
                ((current_chars + q_len > max_chars) or (len(current_batch) >= max_batch_size))
            )
            
            if should_break:
                batch_start = time.time()
                try:
                    batch_results = solver.solve_batch(current_batch, model_type=model, q_type=type_label)
                    batch_end = time.time()
                    
                    # Tính thời gian trung bình cho mỗi câu trong batch
                    # (Đây là cách công bằng nhất cho Pipeline)
                    total_duration = batch_end - batch_start
                    avg_time_per_item = total_duration / len(current_batch)
                    batch_results = solver.solve_batch(current_batch, model_type=model, q_type=type_label)
                    for qid, ans in batch_results.items():
                        results.append({"qid": qid, "answer": ans, "time": avg_time_per_item})
                    
                    
                    save_checkpoint(results)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ User interrupted! Saving progress...")
                    save_checkpoint(results)
                    print("You can resume by running the script again.")
                    sys.exit(0)
                    
                except Exception as e:
                    print(f"\n⚠️ Batch error: {e}")
                    
                    for q_item in current_batch:
                        t_start = time.time()
                        try:
                            if type_label == "MATH":
                                ans = solver.solve_math(q_item['question'], q_item['choices'])
                            elif type_label == "READING":
                                ans = solver.solve_reading(q_item['question'], q_item['choices'])
                            else:
                                ans = solver.solve_knowledge(q_item['question'], q_item['choices'])
                            results.append({"qid": q_item['qid'], "answer": ans,"time": time.time() - t_start})
                        except:
                            results.append({"qid": q_item['qid'], "answer": "A","time": time.time() - t_start})
                    save_checkpoint(results)
                
                pbar.update(len(current_batch))
                
                
                current_batch = []
                current_chars = 0
                time.sleep(1)  # Delay giữa các batch
            
            current_batch.append(q)
            current_chars += q_len
        
        
        if current_batch:
            batch_start = time.time()
            try:
                batch_results = solver.solve_batch(current_batch, model_type=model, q_type=type_label)
                batch_end = time.time()
                
                avg_time_per_item = (batch_end - batch_start) / len(current_batch)
                for qid, ans in batch_results.items():
                    results.append({"qid": qid, "answer": ans,"time": avg_time_per_item})
                save_checkpoint(results)
            except KeyboardInterrupt:
                print("\n⚠️ User interrupted! Saving progress...")
                save_checkpoint(results)
                sys.exit(0)
            except Exception as e:
                print(f"\n⚠️ Final batch error: {e}")
                for q_item in current_batch:
                    t_start = time.time()
                    try:
                        if type_label == "MATH":
                            ans = solver.solve_math(q_item['question'], q_item['choices'])
                        elif type_label == "READING":
                            ans = solver.solve_reading(q_item['question'], q_item['choices'])
                        else:
                            ans = solver.solve_knowledge(q_item['question'], q_item['choices'])
                        results.append({"qid": q_item['qid'], "answer": ans,"time": time.time() - t_start})
                    except:
                        results.append({"qid": q_item['qid'], "answer": "A","time": time.time() - t_start})
                    
                save_checkpoint(results)
            
            pbar.update(len(current_batch))
        
        pbar.close()

    
    
    
    try:
        
        run_smart_batch(buckets["MATH"], "large", max_chars=12000, type_label="MATH")
        
        
        run_smart_batch(buckets["READING"], "large", max_chars=20000, type_label="READING")
        
        
        run_smart_batch(buckets["KNOWLEDGE"], "small", max_chars=50000, type_label="KNOWLEDGE")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user!")
        save_checkpoint(results)
        print("Progress saved. Run again to resume.")
        return

    
    
    
    print("\n🔍 Checking for missing answers...")
    final_ids = set(str(r['qid']) for r in results)
    missing_count = 0
    
    for q in all_questions:
        if str(q['qid']) not in final_ids:
            results.append({"qid": q['qid'], "answer": "A","time": 0.001})
            missing_count += 1
    
    if missing_count > 0:
        print(f"⚠️ Filled {missing_count} missing answers with 'A'")

    
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETED!")
    print(f"   Total questions: {len(df_out)}")
    print(f"   Output saved to: {OUTPUT_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Script terminated by user. Progress has been saved.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        print("Check your setup and try again.")