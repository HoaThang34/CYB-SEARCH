export interface Candidate {
    id: number;
    sbd: string;
    name: string | null;
    province: string;
    school: string;
    subject: string;
    class_grade: string | null;
    
    score_listening: number | null;
    score_speaking: number | null;
    score_reading: number | null;
    score_writing: number | null;
    
    total_score: number | null;
    prize: string | null;
}

export interface RankingItem {
    rank: number;
    data: Candidate;
}
