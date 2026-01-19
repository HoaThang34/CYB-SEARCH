import React, { useEffect, useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { RankingTable } from './components/RankingTable';
import { getRanking, getSubjects, getProvinces, uploadFile } from './services/api';
import { RankingItem, Candidate } from './types';
import { Upload, Crown, Filter } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState<'ranking' | 'search'>('ranking');
  const [subjects, setSubjects] = useState<string[]>([]);
  const [provinces, setProvinces] = useState<string[]>([]);

  const [selectedSubject, setSelectedSubject] = useState<string>('');
  const [selectedProvince, setSelectedProvince] = useState<string>('');
  const [rankings, setRankings] = useState<RankingItem[]>([]);

  // For single result view
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);

  useEffect(() => {
    getSubjects().then(setSubjects);
    getProvinces().then(setProvinces);
  }, []);

  useEffect(() => {
    if (activeTab === 'ranking') {
      loadRanking();
    }
  }, [selectedSubject, selectedProvince, activeTab]);

  const loadRanking = async () => {
    const data = await getRanking(selectedSubject || undefined, selectedProvince || undefined);
    setRankings(data);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      try {
        await uploadFile(e.target.files[0]);
        alert("Upload thành công! Hãy tải lại trang để cập nhật bộ lọc.");
        window.location.reload();
      } catch (error) {
        alert("Upload thất bại.");
      }
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Crown className="h-8 w-8 text-blue-600" />
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              HSGQG 2024
            </h1>
          </div>

          <div className="flex gap-4">
            <button
              onClick={() => setActiveTab('ranking')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === 'ranking' ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'}`}
            >
              Xếp hạng
            </button>
            <button
              onClick={() => setActiveTab('search')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === 'search' ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'}`}
            >
              Tra cứu
            </button>
          </div>

          <div className="relative group">
            <label className="cursor-pointer flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors text-sm font-medium">
              <Upload className="h-4 w-4" />
              <span>Import Data</span>
              <input type="file" className="hidden" accept=".csv" onChange={handleFileUpload} />
            </label>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'search' && (
          <div className="flex flex-col items-center gap-8 fade-in">
            <div className="text-center space-y-2">
              <h2 className="text-3xl font-bold text-slate-900">Tra cứu kết quả thi</h2>
              <p className="text-slate-500">Nhập Số báo danh để tìm kiếm thông tin chi tiết</p>
            </div>
            <SearchBar onSelect={setSelectedCandidate} />

            {selectedCandidate && (
              <div className="w-full max-w-2xl bg-white rounded-xl shadow-lg border border-slate-100 p-8 mt-8">
                <div className="flex justify-between items-start mb-6 border-b border-slate-100 pb-6">
                  <div>
                    <h3 className="text-2xl font-bold text-slate-900">{selectedCandidate.sbd}</h3>
                    <p className="text-slate-500">{selectedCandidate.name || "Chưa cập nhật tên"}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-black text-blue-600">{selectedCandidate.total_score}</div>
                    <div className="text-sm font-medium text-slate-400">Tổng điểm</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-y-4 gap-x-8 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Môn thi</span>
                    <span className="font-semibold">{selectedCandidate.subject}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Đơn vị</span>
                    <span className="font-semibold">{selectedCandidate.province}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Trường</span>
                    <span className="font-semibold">{selectedCandidate.school}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Giải</span>
                    <span className="font-bold text-orange-600">{selectedCandidate.prize || "..."}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'ranking' && (
          <div className="space-y-6 fade-in">
            <div className="flex flex-wrap gap-4 bg-white p-4 rounded-xl shadow-sm border border-slate-200 items-center">
              <Filter className="h-5 w-5 text-slate-400" />
              <select
                className="bg-slate-50 border-none rounded-lg px-4 py-2 font-medium text-slate-700 focus:ring-2 focus:ring-blue-500"
                value={selectedSubject}
                onChange={(e) => setSelectedSubject(e.target.value)}
              >
                <option value="">Tất cả môn thi</option>
                {subjects.map(s => <option key={s} value={s}>{s}</option>)}
              </select>

              <select
                className="bg-slate-50 border-none rounded-lg px-4 py-2 font-medium text-slate-700 focus:ring-2 focus:ring-blue-500"
                value={selectedProvince}
                onChange={(e) => setSelectedProvince(e.target.value)}
              >
                <option value="">Tất cả tỉnh/thành</option>
                {provinces.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>

            <RankingTable items={rankings} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
