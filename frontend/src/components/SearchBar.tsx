import React, { useState, useEffect } from 'react';
import { Search } from 'lucide-react';
import { cn } from '../lib/utils';
import { searchCandidates } from '../services/api';
import { Candidate } from '../types';

interface SearchBarProps {
    onSelect: (candidate: Candidate) => void;
}

export function SearchBar({ onSelect }: SearchBarProps) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<Candidate[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const timer = setTimeout(async () => {
            if (query.trim().length > 2) {
                setLoading(true);
                try {
                    const data = await searchCandidates(query);
                    setResults(data);
                } catch (e) {
                    console.error(e);
                } finally {
                    setLoading(false);
                }
            } else {
                setResults([]);
            }
        }, 500);

        return () => clearTimeout(timer);
    }, [query]);

    return (
        <div className="relative w-full max-w-xl mx-auto">
            <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 h-5 w-5" />
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Nhập SBD..."
                    className={cn(
                        "w-full pl-10 pr-4 py-3 rounded-lg border border-slate-200",
                        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                        "shadow-sm transition-all text-lg"
                    )}
                />
            </div>

            {results.length > 0 && (
                <div className="absolute top-full mt-2 w-full bg-white rounded-lg shadow-xl border border-slate-100 overflow-hidden z-50">
                    <div className="max-h-80 overflow-y-auto">
                        {results.map((c) => (
                            <div
                                key={c.sbd}
                                onClick={() => {
                                    onSelect(c);
                                    setResults([]);
                                    setQuery('');
                                }}
                                className="p-3 hover:bg-slate-50 cursor-pointer border-b border-slate-50 last:border-0"
                            >
                                <div className="font-medium text-slate-900 flex justify-between">
                                    <span>{c.sbd}</span>
                                    <span className="text-blue-600">{c.total_score ?? '-'} đ</span>
                                </div>
                                <div className="text-sm text-slate-500 flex gap-2">
                                    <span>{c.subject}</span>
                                    <span>•</span>
                                    <span>{c.province}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
