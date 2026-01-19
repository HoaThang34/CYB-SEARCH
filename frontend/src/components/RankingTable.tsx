import React, { useState } from 'react';
import { cn } from '../lib/utils';
import { RankingItem } from '../types';
import { Trophy, Medal, Award } from 'lucide-react';

interface RankingTableProps {
    items: RankingItem[];
}

export function RankingTable({ items }: RankingTableProps) {
    if (items.length === 0) {
        return <div className="text-center py-10 text-slate-500">Chưa có dữ liệu xếp hạng</div>;
    }

    return (
        <div className="overflow-x-auto rounded-lg border border-slate-200 shadow-sm">
            <table className="w-full text-left text-sm">
                <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                        <th className="p-4 text-center w-16">Hạng</th>
                        <th className="p-4 w-32">SBD</th>
                        {items[0].data.name && <th className="p-4">Họ tên</th>}
                        <th className="p-4">Đơn vị</th>
                        <th className="p-4">Trường</th>
                        <th className="p-4 text-right">Tổng điểm</th>
                        <th className="p-4">Giải</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                    {items.map((item) => (
                        <tr key={item.data.sbd} className="hover:bg-slate-50/50 transition-colors">
                            <td className="p-4 text-center">
                                <RankBadge rank={item.rank} />
                            </td>
                            <td className="p-4 font-mono text-slate-600">{item.data.sbd}</td>
                            {item.data.name && <td className="p-4 font-medium">{item.data.name}</td>}
                            <td className="p-4">{item.data.province}</td>
                            <td className="p-4 text-slate-500">{item.data.school}</td>
                            <td className="p-4 text-right font-bold text-blue-600">
                                {item.data.total_score}
                            </td>
                            <td className="p-4">
                                <PrizeBadge prize={item.data.prize} />
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function RankBadge({ rank }: { rank: number }) {
    if (rank === 1) return <div className="flex justify-center"><Trophy className="h-5 w-5 text-yellow-500 fill-yellow-500" /></div>;
    if (rank === 2) return <div className="flex justify-center"><Medal className="h-5 w-5 text-gray-400 fill-gray-400" /></div>;
    if (rank === 3) return <div className="flex justify-center"><Medal className="h-5 w-5 text-orange-400 fill-orange-400" /></div>;
    return <span className="font-semibold text-slate-500">#{rank}</span>;
}

function PrizeBadge({ prize }: { prize: string | null }) {
    if (!prize) return null;
    const colors: Record<string, string> = {
        'Nhất': 'bg-red-100 text-red-700 border-red-200',
        'Nhì': 'bg-orange-100 text-orange-700 border-orange-200',
        'Ba': 'bg-yellow-100 text-yellow-700 border-yellow-200',
        'K.Khích': 'bg-blue-100 text-blue-700 border-blue-200',
    };

    // Normalize logic loosely
    let key = Object.keys(colors).find(k => prize.includes(k)) || 'default';
    let className = colors[key] || 'bg-slate-100 text-slate-700 border-slate-200';

    return (
        <span className={cn("px-2 py-1 rounded-full text-xs font-medium border", className)}>
            {prize}
        </span>
    );
}
