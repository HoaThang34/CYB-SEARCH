const API_BASE = '/api';

// Global state to hold current data for modal access
let currentRankingData = [];
let currentSearchData = [];

// Initial Load
document.addEventListener('DOMContentLoaded', async () => {
    setupSidebar();

    const path = window.location.pathname;

    // Inject Generic Modal to DOM
    injectCandidateModal();

    // Always fetch filters if elements exist
    if (document.getElementById('subject-filter')) {
        await fetchFilters();
    }

    // Page specifics
    if (path === '/ranking') {
        loadRanking();
    } else if (path === '/stats') {
        loadStatsPage();
    } else if (path === '/search') {
        // Search page init if needed
    }

    // Custom Cursor
    if (window.matchMedia("(pointer: fine)").matches) {
        initCustomCursor();
    }
});

function injectCandidateModal() {
    if (document.getElementById('candidate-modal')) return;

    const modalHtml = `
    <div id="candidate-modal" class="modal">
        <div class="modal-content glass-panel">
            <div class="modal-header">
                <h3>Chi tiết thí sinh</h3>
                <button class="close-modal-btn" onclick="closeCandidateModal()">&times;</button>
            </div>
            <div class="modal-body" id="candidate-modal-body">
                <!-- Content injected via JS -->
            </div>
        </div>
    </div>`;
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Close on outside click
    window.addEventListener('click', (e) => {
        const modal = document.getElementById('candidate-modal');
        if (e.target === modal) closeCandidateModal();
    });
}

function openCandidateModal(data, source = 'ranking') {
    const modal = document.getElementById('candidate-modal');
    const body = document.getElementById('candidate-modal-body');

    let candidate = data;
    // If index passed
    if (typeof data === 'number') {
        if (source === 'ranking') candidate = currentRankingData[data].data;
        else if (source === 'search') candidate = currentSearchData[data];
    }

    if (!candidate) return;

    const isLanguage = ['Tiếng Anh', 'Tiếng Pháp', 'Tiếng Trung', 'Tiếng Nhật', 'Tiếng Nga', 'Tiếng Hàn', 'Tiếng Đức'].some(lang => candidate.subject.includes(lang));

    let breakdownHtml = '';
    if (isLanguage) {
        breakdownHtml = `
        <div class="scores-breakdown">
            <div class="breakdown-title">Điểm thành phần</div>
            <div class="breakdown-grid">
                <div class="breakdown-item">
                    <span class="breakdown-score">${candidate.score_listening || '-'}</span>
                    <span class="breakdown-label">Nghe</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-score">${candidate.score_speaking || '-'}</span>
                    <span class="breakdown-label">Nói</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-score">${candidate.score_reading || '-'}</span>
                    <span class="breakdown-label">Đọc</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-score">${candidate.score_writing || '-'}</span>
                    <span class="breakdown-label">Viết</span>
                </div>
            </div>
        </div>
        `;
    }

    body.innerHTML = `
        <div class="candidate-detail-card">
            <div class="detail-row main-score">
                <span class="label">Tổng điểm</span>
                <span class="big-score">${candidate.total_score}</span>
                <span class="prize-badge ${getPrizeClass(candidate.prize)}">${candidate.prize || 'Không đạt giải'}</span>
            </div>
            
            <div class="detail-grid">
                <div class="detail-item">
                    <span class="label">Họ và tên</span>
                    <span class="value" style="font-weight:700">${candidate.name || '---'}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Số Báo Danh</span>
                    <span class="value" style="font-family:monospace">${candidate.sbd}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Đơn vị</span>
                    <span class="value">${candidate.province}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Trường</span>
                    <span class="value">${candidate.school}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Môn thi</span>
                    <span class="value">${candidate.subject}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Lớp</span>
                    <span class="value">${candidate.class_grade || '---'}</span>
                </div>
            </div>
            
            ${breakdownHtml}
        </div>
    `;

    modal.classList.add('active');
}

function closeCandidateModal() {
    document.getElementById('candidate-modal').classList.remove('active');
}

function getPrizeClass(prize) {
    if (!prize) return '';
    if (prize.includes('Nhất')) return 'prize-1';
    if (prize.includes('Nhì')) return 'prize-2';
    if (prize.includes('Ba')) return 'prize-3';
    return 'prize-mk';
}

// Sidebar Toggle Logic
function setupSidebar() {
    const toggleBtn = document.getElementById('toggle-sidebar');
    const sidebar = document.getElementById('sidebar');

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }
}

async function fetchFilters() {
    try {
        const [subjRes, provRes] = await Promise.all([
            fetch(`${API_BASE}/subjects`),
            fetch(`${API_BASE}/provinces`)
        ]);

        const subjects = await subjRes.json();
        const provinces = await provRes.json();

        populateSelect('subject-filter', subjects);
        populateSelect('province-filter', provinces);
    } catch (e) {
        console.error("Connection failed", e);
    }
}

function populateSelect(id, items) {
    const el = document.getElementById(id);
    if (!el) return;

    items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        el.appendChild(opt);
    });
}

async function loadRanking() {
    const subjEl = document.getElementById('subject-filter');
    const provEl = document.getElementById('province-filter');

    const subj = subjEl ? subjEl.value : '';
    const prov = provEl ? provEl.value : '';

    let url = `${API_BASE}/ranking?`;
    if (subj) url += `subject=${encodeURIComponent(subj)}&`;
    if (prov) url += `province=${encodeURIComponent(prov)}`;

    const tbody = document.getElementById('ranking-body');
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="7" class="text-center" style="padding:40px">Đang tải dữ liệu...</td></tr>';

    try {
        const res = await fetch(url);
        currentRankingData = await res.json(); // Store for modal (ranking)

        tbody.innerHTML = '';
        if (currentRankingData.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center" style="padding:40px">Không tìm thấy dữ liệu phù hợp</td></tr>';
        } else {
            currentRankingData.forEach((item, idx) => {
                const c = item.data;
                const tr = document.createElement('tr');
                tr.onclick = () => openCandidateModal(idx, 'ranking');
                tr.className = 'clickable-row'; // Add class for styling

                tr.innerHTML = `
                    <td class="text-center"><span class="table-rank-badge">${item.rank}</span></td>
                    <td class="text-center"><span class="sbd-badge" style="background:#f1f5f9; padding:2px 6px; border-radius:4px; font-weight:600; font-size:0.85rem">${c.sbd}</span></td>
                    <td>${c.province}</td>
                    <td style="color:#64748b">${c.school}</td>
                    <td><span style="font-weight:500">${c.subject}</span></td>
                    <td class="text-right"><b style="font-size:1.1rem">${c.total_score}</b></td>
                    <td class="text-center">${renderPrize(c.prize)}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error(e);
        tbody.innerHTML = '<tr><td colspan="7" class="text-center" style="color:red">Lỗi kết nối server</td></tr>';
    }
}

// Search Logic
let searchTimer;
function handleSearch(val) {
    clearTimeout(searchTimer);
    if (!val || val.length < 3) {
        document.getElementById('search-results').innerHTML = '';
        return;
    }

    searchTimer = setTimeout(async () => {
        const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(val)}`);
        currentSearchData = await res.json(); // Store for modal (search)
        const container = document.getElementById('search-results');

        if (!container) return;

        if (currentSearchData.length === 0) {
            container.innerHTML = '<div style="grid-column:1/-1; text-align:center; color:#94a3b8">Không tìm thấy thí sinh nào</div>';
            return;
        }

        container.innerHTML = currentSearchData.map((c, idx) => `
            <div class="result-card clickable-row" onclick="openCandidateModal(${idx}, 'search')">
                <div class="card-top">
                    <div>
                        <div class="sbd-tag">${c.sbd}</div>
                        <div style="margin-top:8px; font-weight:600; font-size:1.1rem">${c.name || 'Thí sinh'}</div>
                    </div>
                    <div class="score-big">${c.total_score}</div>
                </div>
                <div class="info-row">
                    <span class="info-label">Đơn vị</span>
                    <span class="info-val">${c.province}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Trường</span>
                    <span class="info-val">${c.school}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Môn thi</span>
                    <span class="info-val">${c.subject}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Giải</span>
                    <span>${renderPrize(c.prize)}</span>
                </div>
            </div>
        `).join('');
    }, 400);
}

function renderPrize(prize) {
    if (!prize) return '<span style="opacity:0.3">-</span>';
    let cls = getPrizeClass(prize);
    return `<span class="prize-badge ${cls}">${prize}</span>`;
}

// Stats Page & Custom Cursor logic... (Keep existing implementations if needed, simplified below)

async function loadStatsPage() {
    // ... (Your implementation from previous step)
    const subjEl = document.getElementById('subject-filter');
    const subj = subjEl ? subjEl.value : '';

    let statsUrl = `${API_BASE}/stats?`;
    if (subj) {
        statsUrl += `subject=${encodeURIComponent(subj)}`;
    }

    const container = document.getElementById('stats-container-full');
    if (!container) return;

    container.innerHTML = '<div class="stats-card"><p class="text-center">Đang tính toán...</p></div>';

    try {
        const res = await fetch(statsUrl);
        const stats = await res.json();
        renderStatsFull(stats, container);
    } catch (e) {
        console.error(e);
        container.innerHTML = '<div class="stats-card"><p class="text-center text-red-500">Lỗi tải thống kê</p></div>';
    }
}

function renderStatsFull(stats, container) {
    // Overview
    const cardOverview = `
        <div class="stats-card">
            <h3>Tổng Quan</h3>
            <div class="stats-row">
                <span>Tổng thí sinh</span>
                <span>${stats.total_candidates}</span>
            </div>
            <div class="stats-row">
                <span>Có giải</span>
                <span style="color:var(--primary); font-weight:700">${stats.total_prizes} (${stats.prize_ratio}%)</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stats-row"><span>Giải Nhất</span><span class="prize-1" style="padding:2px 8px; font-size:0.75rem">${stats.prizes['Nhất']}</span></div>
            <div class="stats-row"><span>Giải Nhì</span><span class="prize-2" style="padding:2px 8px; font-size:0.75rem">${stats.prizes['Nhì']}</span></div>
            <div class="stats-row"><span>Giải Ba</span><span class="prize-3" style="padding:2px 8px; font-size:0.75rem">${stats.prizes['Ba']}</span></div>
            <div class="stats-row"><span>Khuyến Khích</span><span class="prize-mk" style="padding:2px 8px; font-size:0.75rem">${stats.prizes['K.Khích']}</span></div>
        </div>
    `;

    // Cutoff
    let cardCutoff = '';
    if (Object.keys(stats.cutoff_scores).length > 0) {
        cardCutoff = `
            <div class="stats-card">
                <h3>Điểm Chuẩn (QG)</h3>
                <div class="stats-row"><span>Giải Nhất</span><span>${stats.cutoff_scores['Nhất'] || '-'}</span></div>
                <div class="stats-row"><span>Giải Nhì</span><span>${stats.cutoff_scores['Nhì'] || '-'}</span></div>
                <div class="stats-row"><span>Giải Ba</span><span>${stats.cutoff_scores['Ba'] || '-'}</span></div>
                <div class="stats-row"><span>Khuyến Khích</span><span>${stats.cutoff_scores['K.Khích'] || '-'}</span></div>
                <p style="margin-top:10px; font-size:0.8rem; color:var(--text-light)">* Điểm chuẩn tính trên phạm vi toàn quốc cho môn học này.</p>
            </div>
        `;
    } else {
        cardCutoff = `
            <div class="stats-card">
                <h3>Điểm Chuẩn</h3>
                <p class="text-center" style="color:var(--text-light); padding:20px 0;">Chọn một môn thi cụ thể để xem điểm chuẩn.</p>
            </div>
        `;
    }

    // Top Schools Highlights (Cards)
    let cardSchools = '';
    if (stats.top_schools.length > 0) {
        let listHtml = '';
        stats.top_schools.forEach(s => {
            listHtml += `
                <div class="school-item clickable-school" onclick="showSchool('${s.name}')">
                    <span class="school-name" title="${s.name}">${s.name}</span>
                    <span class="school-count">${s.count}</span>
                </div>
            `;
        });
        cardSchools = `
            <div class="stats-card">
                <h3>Top Trường (Nổi bật)</h3>
                ${listHtml}
            </div>
        `;
    }

    container.innerHTML = cardOverview + cardCutoff + cardSchools;

    // Full School Ranking Table
    if (stats.all_schools && stats.all_schools.length > 0) {
        let tableRows = '';
        stats.all_schools.forEach((s, idx) => {
            tableRows += `
                <tr class="clickable-school" onclick="showSchool('${s.name}')">
                    <td class="text-center"><span class="table-rank-badge" style="width:24px; height:24px; font-size:0.8rem">${idx + 1}</span></td>
                    <td style="font-weight:500">${s.name}</td>
                    <td class="text-center"><span class="prize-1" style="font-size:0.9rem; font-weight:700">${s.details['Nhất'] || '-'}</span></td>
                    <td class="text-center"><span class="prize-2" style="font-size:0.9rem; font-weight:700">${s.details['Nhì'] || '-'}</span></td>
                    <td class="text-center"><span class="prize-3" style="font-size:0.9rem; font-weight:700">${s.details['Ba'] || '-'}</span></td>
                    <td class="text-center"><span class="prize-mk" style="font-size:0.9rem; font-weight:700">${s.details['K.Khích'] || '-'}</span></td>
                    <td class="text-center"><span style="background:var(--primary); color:white; padding:2px 8px; border-radius:12px; font-size:0.85rem; font-weight:600">${s.count}</span></td>
                </tr>
            `;
        });

        const fullTableHtml = `
            <div class="glass-panel" style="grid-column: 1 / -1; margin-top: 20px; padding: 20px;">
                <h3 style="margin-bottom: 20px; font-size: 1.25rem; font-weight: 600; color: var(--text-dark);">Bảng Xếp Hạng Trường (Chi tiết)</h3>
                <div class="table-wrapper" style="max-height: 500px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th class="text-center" width="80px">Hạng</th>
                                <th>Trường</th>
                                <th class="text-center" width="80px">Nhất</th>
                                <th class="text-center" width="80px">Nhì</th>
                                <th class="text-center" width="80px">Ba</th>
                                <th class="text-center" width="80px">KK</th>
                                <th class="text-center" width="100px">Tổng giải</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${tableRows}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        container.innerHTML += fullTableHtml;
    }
}

// School Modal (Keep independent or merge? Independent is fine)
async function showSchool(schoolName) {
    // Check if modal exists in DOM (stats.html has it, others don't)
    // If not, we might need to inject it or just return
    let modal = document.getElementById('school-modal');
    if (!modal) {
        // Inject if missing (for other pages if we want school details there)
        const modalHtml = `
        <div id="school-modal" class="modal">
            <div class="modal-content glass-panel">
                <div class="modal-header">
                    <h3 id="modal-school-name">Tên Trường</h3>
                    <span class="close-btn" onclick="document.getElementById('school-modal').classList.remove('active')">&times;</span>
                </div>
                <div class="modal-body">
                    <table>
                        <thead>
                            <tr>
                                <th width="10%">#</th>
                                <th>SBD</th>
                                <th>Tên</th>
                                <th>Môn</th>
                                <th>Điểm</th>
                                <th>Giải</th>
                            </tr>
                        </thead>
                        <tbody id="modal-school-body"></tbody>
                    </table>
                </div>
            </div>
        </div>`;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        modal = document.getElementById('school-modal');
    }

    const title = document.getElementById('modal-school-name');
    const tbody = document.getElementById('modal-school-body');

    title.innerText = schoolName;
    tbody.innerHTML = '<tr><td colspan="6" class="text-center" style="padding:20px">Đang tải dữ liệu...</td></tr>';
    modal.classList.add('active');

    // Close logic
    window.onclick = function (event) {
        if (event.target == modal) {
            modal.classList.remove('active');
        }
    }

    // Fetch data
    let url = `${API_BASE}/ranking?school=${encodeURIComponent(schoolName)}`;
    try {
        const res = await fetch(url);
        const data = await res.json();

        tbody.innerHTML = '';
        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center" style="padding:20px">Không có dữ liệu</td></tr>';
        } else {
            data.forEach((item, idx) => {
                const c = item.data;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="text-center">${idx + 1}</td>
                    <td class="text-center"><span class="sbd-badge" style="background:#f1f5f9; padding:2px 6px; border-radius:4px; font-weight:600; font-size:0.85rem">${c.sbd}</span></td>
                    <td>${c.name || 'Thí sinh'}</td>
                    <td>${c.subject}</td>
                    <td class="text-right"><b style="font-size:1.1rem">${c.total_score}</b></td>
                    <td class="text-center">${renderPrize(c.prize)}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error(e);
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-red-500">Lỗi kết nối</td></tr>';
    }
}

// Client-side quick filter for table
function filterTable(val) {
    val = val.toLowerCase();
    const rows = document.querySelectorAll('#ranking-body tr');
    rows.forEach(row => {
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(val) ? '' : 'none';
    });
}

function initCustomCursor() {
    if (document.querySelector('.cursor-dot')) return;

    const dot = document.createElement('div');
    dot.className = 'cursor-dot';
    const outline = document.createElement('div');
    outline.className = 'cursor-outline';

    document.body.appendChild(dot);
    document.body.appendChild(outline);

    let cursorX = 0, cursorY = 0;
    let outlineX = 0, outlineY = 0;

    window.addEventListener('mousemove', (e) => {
        cursorX = e.clientX;
        cursorY = e.clientY;
        dot.style.left = `${cursorX}px`;
        dot.style.top = `${cursorY}px`;
        dot.style.opacity = 1;
        outline.style.opacity = 0.5;
    });

    function animateOutline() {
        outlineX += (cursorX - outlineX) * 0.15;
        outlineY += (cursorY - outlineY) * 0.15;
        outline.style.left = `${outlineX}px`;
        outline.style.top = `${outlineY}px`;
        requestAnimationFrame(animateOutline);
    }
    animateOutline();

    const hoverSelectors = 'a, button, input, select, textarea, .clickable-school, .nav-item, .feature-card, .btn, label, .clickable-row, .close-modal-btn';

    document.addEventListener('mouseover', (e) => {
        if (e.target.closest(hoverSelectors)) {
            document.body.classList.add('hovering');
        }
    });

    document.addEventListener('mouseout', (e) => {
        if (e.target.closest(hoverSelectors)) {
            document.body.classList.remove('hovering');
        }
    });
}
