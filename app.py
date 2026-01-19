
import os
import shutil
import pandas as pd
from typing import List, Optional

from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sqlalchemy import create_engine, Column, Integer, String, Float, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ---------------------------------------------------------
# 1. DATABASE SETUP
# ---------------------------------------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./hsgqg.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------
# 2. MODELS
# ---------------------------------------------------------
class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    sbd = Column(String, unique=True, index=True)  # Số báo danh
    name = Column(String, nullable=True)          # Họ và tên
    province = Column(String, index=True)         # Đơn vị (Tỉnh/Thành)
    school = Column(String, index=True)           # Đơn vị (Trường)
    subject = Column(String, index=True)          # Môn thi
    
    # Detail Scores
    score_listening = Column(Float, nullable=True)
    score_speaking = Column(Float, nullable=True)
    score_reading = Column(Float, nullable=True)
    score_writing = Column(Float, nullable=True)
    
    total_score = Column(Float, index=True)       # Tổng điểm
    prize = Column(String, nullable=True)         # Giải
    class_grade = Column(String, nullable=True)   # Lớp

# ---------------------------------------------------------
# 3. HELPER LOGIC (Ranking, Stats)
# ---------------------------------------------------------
PRIZE_ORDER = {
    "Nhất": 4, "Nhì": 3, "Ba": 2, "K.Khích": 1, 
    "Không giải": 0, None: 0
}

def get_prize_value(prize_str):
    return PRIZE_ORDER.get(prize_str, 0)

def get_ranking(db: Session, subject: str = None, province: str = None, school: str = None):
    query = db.query(Candidate)
    if subject:
        query = query.filter(Candidate.subject == subject)
    
    candidates = query.all()
    
    # Group by subject
    candidates_by_subject = {}
    for c in candidates:
        if c.subject not in candidates_by_subject:
            candidates_by_subject[c.subject] = []
        candidates_by_subject[c.subject].append(c)
        
    results_with_rank = []
    
    for subj, sub_candidates in candidates_by_subject.items():
        # Sort
        sub_candidates.sort(key=lambda c: (
            c.total_score if c.total_score is not None else -1,
            get_prize_value(c.prize)
        ), reverse=True)
        
        current_rank = 1
        for i, candidate in enumerate(sub_candidates):
            if i > 0:
                prev = sub_candidates[i-1]
                if (candidate.total_score == prev.total_score and 
                    get_prize_value(candidate.prize) == get_prize_value(prev.prize)):
                    pass 
                else:
                    current_rank = i + 1
            
            # Filters
            if province and candidate.province != province:
                continue
            if school and candidate.school != school:
                continue
                
            results_with_rank.append({
                "rank": current_rank,
                "data": candidate
            })
            
    results_with_rank.sort(key=lambda x: x["rank"])
    return results_with_rank

def get_statistics(db: Session, subject: str = None, province: str = None):
    query = db.query(Candidate)
    if subject:
        query = query.filter(Candidate.subject == subject)
    if province:
        query = query.filter(Candidate.province == province)
        
    candidates = query.all()
    total_candidates = len(candidates)
    
    prizes = {"Nhất": 0, "Nhì": 0, "Ba": 0, "K.Khích": 0}
    total_prizes = 0
    
    for c in candidates:
        if c.prize in prizes:
            prizes[c.prize] += 1
            total_prizes += 1
            
    prize_ratio = 0
    if total_candidates > 0:
        prize_ratio = round((total_prizes / total_candidates) * 100, 2)
        
    # Cutoff scores (National)
    cutoff_scores = {}
    if subject:
        national_query = db.query(Candidate).filter(Candidate.subject == subject).all()
        min_scores = {}
        for c in national_query:
            if c.prize in prizes and c.total_score is not None:
                if c.prize not in min_scores:
                    min_scores[c.prize] = c.total_score
                else:
                    min_scores[c.prize] = min(min_scores[c.prize], c.total_score)
        cutoff_scores = min_scores

    # School Stats
    school_stats = {}
    for c in candidates:
        if c.prize in prizes:
            s_name = c.school if c.school else "Unknown"
            if s_name not in school_stats:
                school_stats[s_name] = {"total": 0, "Nhất": 0, "Nhì": 0, "Ba": 0, "K.Khích": 0}
            school_stats[s_name][c.prize] += 1
            school_stats[s_name]["total"] += 1
            
    all_schools_list = []
    for name, stats in school_stats.items():
        all_schools_list.append({
            "name": name,
            "count": stats["total"],
            "details": stats
        })
        
    all_schools_list.sort(key=lambda s: (
        s["count"], 
        s["details"]["Nhất"], 
        s["details"]["Nhì"], 
        s["details"]["Ba"]
    ), reverse=True)
    
    return {
        "total_candidates": total_candidates,
        "total_prizes": total_prizes,
        "prize_ratio": prize_ratio,
        "prizes": prizes,
        "cutoff_scores": cutoff_scores,
        "top_schools": all_schools_list[:5],
        "all_schools": all_schools_list
    }

# ---------------------------------------------------------
# 4. IMPORT LOGIC
# ---------------------------------------------------------
def import_csv(file_path: str):
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    
    try:
        # FULL RESET for consistency as requested
        session.query(Candidate).delete()
        session.commit()
        
        # Read with naive pandas read_csv
        # Use dtype=str for IDs to preserve leading zeros
        # Format expected: sbd, đơn vị, trường, môn, điểm, giải
        df = pd.read_csv(file_path, dtype={'sbd': str})
        
        # Helper to safely get float
        def get_float(val):
            try:
                if pd.isna(val) or str(val).strip() == '':
                    return None
                val_str = str(val).replace(',', '.') # Handle comma decimal separator if present
                return float(val_str)
            except:
                return None

        # Helper to safely get string or None (never 'nan')
        def get_str(val):
            if pd.isna(val) or str(val).strip().lower() == 'nan' or str(val).strip() == '':
                return None
            return str(val).strip()

        candidates = []
        # Normalize column names to handle potential casing issues
        df.columns = [c.strip().lower() for c in df.columns]

        for _, row in df.iterrows():
            cand = Candidate(
                sbd=get_str(row.get('sbd')),
                name=None,
                # Support both requested format (đơn vị) and file format (don_vi)
                province=get_str(row.get('đơn vị') or row.get('don_vi')),
                school=get_str(row.get('trường') or row.get('truong')),
                subject=get_str(row.get('môn') or row.get('mon')),
                class_grade=None,
                
                # Detailed scores are not present in the new format
                score_writing=None,
                score_listening=None,
                score_reading=None,
                score_speaking=None,
                
                total_score=get_float(row.get('điểm') or row.get('diem')),
                prize=get_str(row.get('giải') or row.get('giai'))
            )
            candidates.append(cand)
            
        session.add_all(candidates)
        session.commit()
        print(f"Imported {len(candidates)} records successfully.")
        
    except Exception as e:
        session.rollback()
        print(f"Import error: {e}")
    finally:
        session.close()

# ---------------------------------------------------------
# 5. FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="HSGQG System")

# Mount Static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    
    # Auto-import if CSV exists
    csv_path = "Ket qua hsg quoc gia.csv"
    if os.path.exists(csv_path):
        print(f"Found {csv_path}, running import...")
        import_csv(csv_path)

@app.get("/")
async def read_home():
    return FileResponse('static/home.html')

@app.get("/ranking")
async def read_ranking_page():
    return FileResponse('static/ranking.html')

@app.get("/search")
async def read_search_page():
    return FileResponse('static/search.html')

@app.get("/stats")
async def read_stats_page():
    return FileResponse('static/stats.html')

# API
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"uploaded_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    import_csv(file_location)
    return {"info": f"File '{file.filename}' imported successfully"}

@app.get("/api/ranking")
def api_ranking(subject: str = None, province: str = None, school: str = None, db: Session = Depends(get_db)):
    return get_ranking(db, subject, province, school)

@app.get("/api/stats")
def api_stats(subject: str = None, province: str = None, db: Session = Depends(get_db)):
    return get_statistics(db, subject, province)

@app.get("/api/search")
def api_search(q: str, db: Session = Depends(get_db)):
    return db.query(Candidate).filter(Candidate.sbd.contains(q)).all()

@app.get("/api/subjects")
def get_subjects(db: Session = Depends(get_db)):
    return [r[0] for r in db.query(Candidate.subject).distinct().order_by(Candidate.subject).all() if r[0]]

@app.get("/api/provinces")
def get_provinces(db: Session = Depends(get_db)):
    return [r[0] for r in db.query(Candidate.province).distinct().order_by(Candidate.province).all() if r[0]]

if __name__ == "__main__":
    import uvicorn
    # Clean up common garbage before running
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__", ignore_errors=True)
        
    print("Starting server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
