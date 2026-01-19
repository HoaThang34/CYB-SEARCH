import axios from 'axios';
import { Candidate, RankingItem } from '../types';

const API_URL = 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_URL,
});

export const searchCandidates = async (query: string): Promise<Candidate[]> => {
    const response = await api.get<Candidate[]>(`/search`, { params: { q: query } });
    return response.data;
};

export const getRanking = async (subject?: string, province?: string): Promise<RankingItem[]> => {
    const response = await api.get<RankingItem[]>(`/ranking`, {
        params: { subject, province }
    });
    return response.data;
};

export const getSubjects = async (): Promise<string[]> => {
    const response = await api.get<string[]>('/subjects');
    return response.data;
};

export const getProvinces = async (): Promise<string[]> => {
    const response = await api.get<string[]>('/provinces');
    return response.data;
};

export const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};
