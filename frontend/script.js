/**
 * GOOGLE GEMINI CLINICAL AI CLIENT LOGIC
 * Real-Time Streaming (SSE) & Progressive Typewriter Effect
 */

document.addEventListener("DOMContentLoaded", () => {
    initGeminiApp();
});

const state = {
    config: null,
    isGenerating: false,
    explorerData: [],
    activeMenuChatId: null,
    currentViewingChatId: null,
    currentTheme: localStorage.getItem("gemini_theme") || "dark",
    isSidebarMini: localStorage.getItem("gemini_sidebar_mini") === "true",
    recentChats: [
        { id: "chat-1", title: "Hướng dẫn hoàn thiện bài báo khoa h...", query: "Hướng dẫn hoàn thiện bài báo khoa học MedKG-HRR và ICD-10", isPinned: false },
        { id: "chat-2", title: "Hướng dẫn Prompt Claude Tìm Dữ Liệu", query: "Hướng dẫn Prompt Claude Tìm Dữ Liệu và mã nguồn thí nghiệm", isPinned: false },
        { id: "chat-3", title: "AI Agent Harness Pipeline Protocol", query: "AI Agent Harness Pipeline Protocol phân tích triệu chứng y khoa", isPinned: false }
    ]
};

const GEMINI_SPARKLE_SVG = `
<svg class="gemini-sparkle-avatar" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M14 0C14 7.73199 7.73199 14 0 14C7.73199 14 14 20.268 14 28C14 20.268 20.268 14 28 14C20.268 14 14 7.73199 14 0Z" fill="url(#star-grad-resp)"/>
    <defs>
        <linearGradient id="star-grad-resp" x1="0" y1="0" x2="28" y2="28" gradientUnits="userSpaceOnUse">
            <stop stop-color="#ffffff"/>
            <stop offset="0.35" stop-color="#c7c7c7ff"/>
            <stop offset="0.7" stop-color="#c7c7c7ff"/>
            <stop offset="1" stop-color="#c7c7c7ff"/>
        </linearGradient>
    </defs>
</svg>
`;

function initGeminiApp() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.getVoices();
        window.speechSynthesis.onvoiceschanged = () => {
            window.speechSynthesis.getVoices();
        };
    }
    applyTheme(state.currentTheme);
    applySidebarState();
    loadSystemStatus();
    loadSystemConfig();
    renderRecentChatsList();
    setupEventListeners();
    setupContextMenuActions();
    setupSettingsPopoverActions();
    setupTextarea();
}

// ============================================================
// SIDEBAR EXPANDED <-> MINI RAIL (68px)
// ============================================================
function applySidebarState() {
    const sidebar = document.getElementById("sidebar");
    if (sidebar) {
        if (state.isSidebarMini) {
            sidebar.classList.add("mini-rail");
        } else {
            sidebar.classList.remove("mini-rail");
        }
    }
}

function toggleSidebarMode() {
    state.isSidebarMini = !state.isSidebarMini;
    localStorage.setItem("gemini_sidebar_mini", state.isSidebarMini);
    applySidebarState();
}

window.handleLogoClick = function () {
    if (state.isSidebarMini) {
        toggleSidebarMode();
    }
};

// ============================================================
// THEME MANAGEMENT (HỆ THỐNG / SÁNG / TỐI)
// ============================================================
window.toggleThemeSubmenu = function (e) {
    if (e) e.stopPropagation();
    const flyout = document.getElementById("theme-flyout-menu");
    if (flyout) {
        flyout.style.display = flyout.style.display === "none" ? "flex" : "none";
    }
};

window.selectTheme = function (theme, e) {
    if (e) e.stopPropagation();
    state.currentTheme = theme;
    localStorage.setItem("gemini_theme", theme);
    applyTheme(theme);

    const flyout = document.getElementById("theme-flyout-menu");
    if (flyout) flyout.style.display = "none";
    hideSettingsPopover();

    const themeNames = { system: "Hệ thống", light: "Sáng", dark: "Tối" };
    showToast(`Đã chuyển sang giao diện ${themeNames[theme] || theme}`);
};

function applyTheme(theme) {
    let activeTheme = theme;
    if (theme === "system") {
        const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        activeTheme = prefersDark ? "dark" : "light";
    }

    document.documentElement.setAttribute("data-theme", activeTheme);

    const icons = {
        system: document.querySelector("#opt-theme-system .theme-check-icon"),
        light: document.querySelector("#opt-theme-light .theme-check-icon"),
        dark: document.querySelector("#opt-theme-dark .theme-check-icon")
    };

    Object.keys(icons).forEach(k => {
        if (icons[k]) {
            icons[k].style.display = (k === theme) ? "inline-block" : "none";
        }
    });
}

// ============================================================
// RECENT CHATS RENDERING
// ============================================================
function renderRecentChatsList() {
    const container = document.getElementById("recent-list-container");
    if (!container) return;

    if (state.recentChats.length === 0) {
        container.innerHTML = `<div style="padding: 10px 12px; font-size: 0.8rem; color: #8e918f;">Chưa có cuộc trò chuyện nào</div>`;
        return;
    }

    container.innerHTML = "";
    state.recentChats.forEach((chat) => {
        const row = document.createElement("div");
        row.className = `recent-chat-row ${state.currentViewingChatId === chat.id ? 'active' : ''}`;
        row.id = `recent-row-${chat.id}`;

        const pinIcon = chat.isPinned ? `<i class="fa-solid fa-thumbtack" style="font-size: 0.7rem; color: #fbbc04; margin-right: 6px;"></i>` : '';

        row.innerHTML = `
            <div class="recent-chat-title" onclick="loadRecentChat('${chat.id}')">
                ${pinIcon}<span>${escapeHtml(chat.title)}</span>
            </div>
            <button class="btn-chat-options" onclick="openChatContextMenu(event, '${chat.id}')" title="Tùy chọn">
                <i class="fa-solid fa-ellipsis-vertical"></i>
            </button>
        `;
        container.appendChild(row);
    });
}

window.loadRecentChat = function (chatId) {
    const chat = state.recentChats.find(c => c.id === chatId);
    if (chat) {
        state.currentViewingChatId = chatId;
        document.querySelectorAll(".recent-chat-row").forEach(r => r.classList.remove("active"));
        const activeRow = document.getElementById(`recent-row-${chatId}`);
        if (activeRow) activeRow.classList.add("active");

        document.getElementById("user-input").value = chat.query;
        handleSendMessage(chat.query);
    }
};

// ============================================================
// CONTEXT MENU POPOVER
// ============================================================
window.openChatContextMenu = function (e, chatId) {
    e.stopPropagation();
    hideSettingsPopover();
    state.activeMenuChatId = chatId;

    const menu = document.getElementById("chat-context-menu");
    const chat = state.recentChats.find(c => c.id === chatId);
    if (!chat) return;

    const pinLabel = document.getElementById("label-pin-text");
    if (pinLabel) {
        pinLabel.textContent = chat.isPinned ? "Bỏ ghim" : "Ghim";
    }

    const btn = e.currentTarget;
    const btnRect = btn.getBoundingClientRect();
    const menuHeight = 225;
    const windowHeight = window.innerHeight;
    const spaceBelow = windowHeight - btnRect.bottom;
    const spaceAbove = btnRect.top;

    menu.style.display = "flex";
    const leftPos = Math.min(btnRect.right + 6, window.innerWidth - 250);
    menu.style.left = `${leftPos}px`;

    if (spaceBelow < menuHeight && spaceAbove > spaceBelow) {
        menu.classList.remove("align-down");
        menu.classList.add("align-up");
        const topPos = Math.max(10, btnRect.bottom - menuHeight);
        menu.style.top = `${topPos}px`;
    } else {
        menu.classList.remove("align-up");
        menu.classList.add("align-down");
        const topPos = Math.max(10, Math.min(btnRect.top - 6, windowHeight - menuHeight - 10));
        menu.style.top = `${topPos}px`;
    }

    document.querySelectorAll(".recent-chat-row").forEach(r => r.classList.remove("menu-open"));
    const activeRow = document.getElementById(`recent-row-${chatId}`);
    if (activeRow) activeRow.classList.add("menu-open");
};

function hideContextMenu() {
    const menu = document.getElementById("chat-context-menu");
    if (menu) menu.style.display = "none";
    document.querySelectorAll(".recent-chat-row").forEach(r => r.classList.remove("menu-open"));
    state.activeMenuChatId = null;
}

function setupContextMenuActions() {
    const safeBind = (id, fn) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("click", fn);
    };

    safeBind("menu-opt-share", () => {
        const chat = state.recentChats.find(c => c.id === state.activeMenuChatId);
        hideContextMenu();
        if (chat) {
            const shareUrl = `${window.location.origin}/?chat=${encodeURIComponent(chat.title)}`;
            navigator.clipboard.writeText(shareUrl).then(() => {
                showToast("Đã sao chép liên kết chia sẻ cuộc trò chuyện!");
            });
        }
    });

    safeBind("menu-opt-pin", () => {
        const chat = state.recentChats.find(c => c.id === state.activeMenuChatId);
        hideContextMenu();
        if (chat) {
            chat.isPinned = !chat.isPinned;
            state.recentChats.sort((a, b) => (b.isPinned ? 1 : 0) - (a.isPinned ? 1 : 0));
            renderRecentChatsList();
            showToast(chat.isPinned ? "Đã ghim cuộc trò chuyện" : "Đã bỏ ghim cuộc trò chuyện");
        }
    });

    safeBind("menu-opt-rename", () => {
        const chat = state.recentChats.find(c => c.id === state.activeMenuChatId);
        hideContextMenu();
        if (chat) {
            const newName = prompt("Nhập tên mới cho cuộc trò chuyện:", chat.title);
            if (newName && newName.trim()) {
                chat.title = newName.trim();
                renderRecentChatsList();
                showToast("Đã đổi tên cuộc trò chuyện");
            }
        }
    });

    safeBind("menu-opt-delete", () => {
        const chatId = state.activeMenuChatId;
        hideContextMenu();
        if (chatId) {
            state.recentChats = state.recentChats.filter(c => c.id !== chatId);
            renderRecentChatsList();

            if (state.currentViewingChatId === chatId) {
                document.getElementById("messages-stream").innerHTML = "";
                document.getElementById("center-hero").style.display = "block";
                state.currentViewingChatId = null;
            }
            showToast("Đã xóa cuộc trò chuyện");
        }
    });

    document.addEventListener("click", (e) => {
        if (!e.target.closest("#chat-context-menu") && !e.target.closest(".btn-chat-options")) {
            hideContextMenu();
        }
        if (!e.target.closest("#settings-popover-menu") && !e.target.closest("#btn-settings-gear") && !e.target.closest("#user-profile-btn") && !e.target.closest("#theme-flyout-menu")) {
            hideSettingsPopover();
        }
    });

    const scrollArea = document.getElementById("sidebar-scroll-area");
    if (scrollArea) {
        scrollArea.addEventListener("scroll", hideContextMenu);
    }
}

// ============================================================
// SETTINGS GEAR POPOVER
// ============================================================
window.toggleSettingsPopover = function (e) {
    if (e) {
        e.stopPropagation();
    }
    hideContextMenu();
    const pop = document.getElementById("settings-popover-menu");
    if (!pop) return;
    const isHidden = (pop.style.display === "none" || !pop.style.display);
    pop.style.display = isHidden ? "block" : "none";
    if (!isHidden) {
        const flyout = document.getElementById("theme-flyout-menu");
        if (flyout) flyout.style.display = "none";
    }
};

window.hideSettingsPopover = function () {
    const pop = document.getElementById("settings-popover-menu");
    if (pop) pop.style.display = "none";
    const flyout = document.getElementById("theme-flyout-menu");
    if (flyout) flyout.style.display = "none";
};

function setupSettingsPopoverActions() {
    const profileBtn = document.getElementById("user-profile-btn");
    if (profileBtn) {
        profileBtn.addEventListener("click", (e) => {
            if (e.target.closest("#btn-settings-gear")) {
                return;
            }
            if (state.isSidebarMini) {
                toggleSidebarMode();
            } else {
                window.toggleSettingsPopover(e);
            }
        });
    }

    const safeBind = (id, fn) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("click", fn);
    };

    safeBind("set-opt-activity", () => {
        hideSettingsPopover();
        showToast("Hệ thống: 8.214 Chunks • 8.214 Vector FAISS • 1.109 Bệnh ICD-10 trực tuyến");
    });

    safeBind("set-opt-personal-ai", () => {
        hideSettingsPopover();
        openSettingsModal();
    });

    safeBind("set-opt-icd-data", () => {
        hideSettingsPopover();
        openExplorerModal();
    });

    safeBind("set-opt-quota", () => {
        hideSettingsPopover();
        showToast("Hạn mức: RAG Hybrid (FAISS IndexFlatIP + BM25) không giới hạn truy vấn");
    });

    safeBind("set-opt-medkg-paper", () => {
        hideSettingsPopover();
        showToast("Hành động đã lên lịch: Đang đồng bộ hóa đồ thị tri thức");
    });

    safeBind("set-opt-skills", () => {
        hideSettingsPopover();
        showToast("Encoder: PhoBERT Contrastive 768 chiều • Reranker: Reciprocal Rank Fusion k=60");
    });

    safeBind("set-opt-knowledge-graph", () => {
        hideSettingsPopover();
        showToast("Neo4j Aura: Đồ thị tri thức thế hệ B (BenhLy, TrieuChung, ICD10, ChuyenKhoa, XetNghiem)");
    });

    safeBind("set-opt-share-link", () => {
        hideSettingsPopover();
        navigator.clipboard.writeText(window.location.href).then(() => {
            showToast("Đã sao chép liên kết hệ thống chẩn đoán MedKG-HRR!");
        });
    });

    safeBind("set-opt-spark", () => {
        hideSettingsPopover();
        const tabSpark = document.getElementById("tab-spark");
        if (tabSpark) tabSpark.click();
    });

    safeBind("set-opt-subscription", () => {
        hideSettingsPopover();
        showToast("Gói: Pro Bác sĩ Lâm sàng (Bá Thành Huỳnh - Quyền truy cập không giới hạn)");
    });

    safeBind("set-opt-ultra", () => {
        hideSettingsPopover();
        showToast("Google AI Ultra: Kích hoạt phiên bản MedKG-HRR Enterprise");
    });

    safeBind("set-opt-feedback", () => {
        hideSettingsPopover();
        const fb = prompt("Nhập ý kiến đóng góp hoặc phản hồi chẩn đoán lâm sàng của bạn:");
        if (fb && fb.trim()) {
            showToast("Cảm ơn bạn! Ý kiến phản hồi đã được gửi đến ban phát triển MedKG.");
        }
    });

    safeBind("set-opt-help", () => {
        hideSettingsPopover();
        showToast("Trợ giúp: Hệ thống MedKG-HRR tuân thủ Luật Khám bệnh, chữa bệnh số 15/2023/QH15");
    });

    // ============================================================
    // TỰ ĐỘNG MỞ MENU GIAO DIỆN KHI RÊ CHUỘT (HOVER) HOẶC CLICK
    // ============================================================
    const themeItem = document.getElementById("set-opt-theme");
    const themeFlyout = document.getElementById("theme-flyout-menu");

    if (themeItem && themeFlyout) {
        let hideTimer = null;

        const showFlyout = (e) => {
            if (e) e.stopPropagation();
            clearTimeout(hideTimer);
            const rect = themeItem.getBoundingClientRect();
            themeFlyout.style.display = "flex";
            themeFlyout.style.position = "fixed";
            themeFlyout.style.left = `${rect.right + 6}px`;
            themeFlyout.style.top = `${rect.top - 4}px`;
            themeFlyout.style.zIndex = "99999";
        };

        const scheduleHide = () => {
            hideTimer = setTimeout(() => {
                themeFlyout.style.display = "none";
            }, 200);
        };

        themeItem.addEventListener("mouseenter", showFlyout);
        themeItem.addEventListener("mouseleave", scheduleHide);

        themeItem.addEventListener("click", (e) => {
            e.stopPropagation();
            if (themeFlyout.style.display === "none" || !themeFlyout.style.display) {
                showFlyout(e);
            } else {
                themeFlyout.style.display = "none";
            }
        });

        themeFlyout.addEventListener("mouseenter", () => clearTimeout(hideTimer));
        themeFlyout.addEventListener("mouseleave", scheduleHide);
    }
}

// ============================================================
// MAIN EVENT LISTENERS
// ============================================================
function setupEventListeners() {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("btn-send");
    const micBtn = document.getElementById("btn-capsule-mic");

    if (userInput && sendBtn && micBtn) {
        userInput.addEventListener("input", function () {
            if (this.value.trim().length > 0) {
                sendBtn.style.display = "flex";
                micBtn.style.display = "none";
            } else {
                sendBtn.style.display = "none";
                micBtn.style.display = "flex";
            }
        });

        userInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    }

    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const text = userInput.value.trim();
            if (text && !state.isGenerating) {
                handleSendMessage(text);
            }
        });
    }

    // TOGGLE SIDEBAR
    const btnToggle = document.getElementById("btn-toggle-sidebar");
    if (btnToggle) btnToggle.addEventListener("click", toggleSidebarMode);

    // Reset Chat
    const resetChat = () => {
        document.getElementById("messages-stream").innerHTML = "";
        document.getElementById("center-hero").style.display = "block";
        if (userInput) {
            userInput.value = "";
            userInput.style.height = "auto";
        }
        if (sendBtn) sendBtn.style.display = "none";
        if (micBtn) micBtn.style.display = "flex";
        state.currentViewingChatId = null;
        document.querySelectorAll(".recent-chat-row").forEach(r => r.classList.remove("active"));
        showToast("Đã bắt đầu cuộc trò chuyện mới");
    };

    const btnNewChat = document.getElementById("btn-new-chat");
    if (btnNewChat) btnNewChat.addEventListener("click", resetChat);

    const btnTopNew = document.getElementById("btn-top-new");
    if (btnTopNew) btnTopNew.addEventListener("click", resetChat);

    // Search button
    const btnSearch = document.getElementById("btn-search-chats");
    if (btnSearch) btnSearch.addEventListener("click", openExplorerModal);

    if (micBtn) micBtn.addEventListener("click", handleVoiceInput);

    const btnCapsulePlus = document.getElementById("btn-capsule-plus");
    if (btnCapsulePlus) btnCapsulePlus.addEventListener("click", openExplorerModal);

    const btnModelPill = document.getElementById("btn-model-pill");
    if (btnModelPill) btnModelPill.addEventListener("click", openSettingsModal);

    document.querySelectorAll(".gemini-modal-overlay").forEach((modal) => {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.classList.remove("open");
        });
    });

    const settingsForm = document.getElementById("settings-form");
    if (settingsForm) settingsForm.addEventListener("submit", handleSaveSettings);

    const btnToggleKey = document.getElementById("btn-toggle-key-visibility");
    if (btnToggleKey) btnToggleKey.addEventListener("click", toggleKeyVisibility);

    const btnTestLLM = document.getElementById("btn-test-llm");
    if (btnTestLLM) btnTestLLM.addEventListener("click", handleTestLLMConnection);

    const tempSlider = document.getElementById("input-temperature");
    const tempVal = document.getElementById("val-temperature");
    if (tempSlider && tempVal) {
        tempSlider.addEventListener("input", () => { tempVal.textContent = tempSlider.value; });
    }

    const topKSlider = document.getElementById("input-top-k");
    const topKVal = document.getElementById("val-top-k");
    if (topKSlider && topKVal) {
        topKSlider.addEventListener("input", () => { topKVal.textContent = topKSlider.value; });
    }

    const expSearch = document.getElementById("explorer-search-input");
    if (expSearch) {
        let expTimer = null;
        expSearch.addEventListener("input", (e) => {
            clearTimeout(expTimer);
            expTimer = setTimeout(() => filterExplorer(e.target.value), 200);
        });
    }
}

function handleVoiceInput() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showToast("Trình duyệt không hỗ trợ nhận diện giọng nói");
        return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'vi-VN';
    recognition.interimResults = false;

    showToast("Đang lắng nghe... Hãy nói triệu chứng");
    recognition.start();

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        const input = document.getElementById("user-input");
        input.value = transcript;
        document.getElementById("btn-send").style.display = "flex";
        document.getElementById("btn-capsule-mic").style.display = "none";
        input.focus();
    };

    recognition.onerror = () => {
        showToast("Không thể nhận diện giọng nói.");
    };
}

function setupTextarea() {
    const textarea = document.getElementById("user-input");
    if (textarea) {
        textarea.addEventListener("input", function () {
            this.style.height = "auto";
            this.style.height = Math.min(this.scrollHeight, 140) + "px";
        });
    }
}

// ============================================================
// API STATUS & CONFIG
// ============================================================
async function loadSystemStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        if (data.status === "online") {
            const pillModel = document.getElementById("current-pill-model-name");
            if (pillModel && data.llm?.current_model) {
                if (data.llm.current_model.includes("flash")) {
                    pillModel.textContent = "Flash";
                } else if (data.llm.current_model.includes("pro")) {
                    pillModel.textContent = "Pro";
                } else {
                    pillModel.textContent = "MedKG-HRR";
                }
            }
        }
    } catch (e) {
        console.warn("Status err:", e);
    }
}

async function loadSystemConfig() {
    try {
        const res = await fetch("/api/config");
        const data = await res.json();
        state.config = data;

        if (data.gemini_model) {
            const el = document.getElementById("select-gemini-model");
            if (el) el.value = data.gemini_model;
        }
        if (data.temperature !== undefined) {
            const el = document.getElementById("input-temperature");
            if (el) el.value = data.temperature;
            const elVal = document.getElementById("val-temperature");
            if (elVal) elVal.textContent = data.temperature;
        }
        if (data.top_k !== undefined) {
            const el = document.getElementById("input-top-k");
            if (el) el.value = data.top_k;
            const elVal = document.getElementById("val-top-k");
            if (elVal) elVal.textContent = data.top_k;
        }
        if (data.gemini_api_key_set) {
            const el = document.getElementById("input-gemini-key");
            if (el) el.placeholder = "•••••••••••••••••••••••••••••••• (Đã cấu hình)";
        }
    } catch (e) {
        console.warn("Config err:", e);
    }
}

// ============================================================
// REAL-TIME STREAMING CHAT (SSE & TYPEWRITER)
// ============================================================
async function handleSendMessage(queryText) {
    document.getElementById("center-hero").style.display = "none";

    const stream = document.getElementById("messages-stream");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("btn-send");
    const micBtn = document.getElementById("btn-capsule-mic");

    const existing = state.recentChats.find(c => c.query === queryText);
    let newChatId = existing ? existing.id : null;
    if (!existing) {
        newChatId = "chat-" + Date.now();
        const shortTitle = queryText.length > 32 ? queryText.substring(0, 32) + "..." : queryText;
        state.recentChats.unshift({
            id: newChatId,
            title: shortTitle,
            query: queryText,
            isPinned: false
        });
        renderRecentChatsList();
    }
    state.currentViewingChatId = newChatId;

    // 1. Render User Bubble
    const userRow = document.createElement("div");
    userRow.className = "gemini-msg-row user-row";
    userRow.innerHTML = `<div class="gemini-user-bubble">${escapeHtml(queryText)}</div>`;
    stream.appendChild(userRow);

    // 2. Tạo ngay bong bóng AI với hiệu ứng 3 chấm suy nghĩ (Gemini Thinking Loader)
    const msgId = "medkg-msg-" + Date.now();
    const aiRow = document.createElement("div");
    aiRow.className = "gemini-msg-row ai-row";
    aiRow.innerHTML = `
        <div class="gemini-ai-body" id="${msgId}">
            <div class="gemini-markdown-content text-content">
                <div class="gemini-thinking-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
            <div class="gemini-candidates-container" id="cand-${msgId}" style="display: none;"></div>
            <div class="gemini-action-bar" id="action-${msgId}" style="display: none;"></div>
        </div>
    `;
    stream.appendChild(aiRow);
    scrollViewport();

    if (userInput) {
        userInput.value = "";
        userInput.style.height = "auto";
    }
    if (sendBtn) sendBtn.style.display = "none";
    if (micBtn) micBtn.style.display = "flex";
    state.isGenerating = true;

    const textContentEl = aiRow.querySelector(".text-content");
    const candContainerEl = document.getElementById(`cand-${msgId}`);
    const actionBarEl = document.getElementById(`action-${msgId}`);

    let accumulatedText = "";

    try {
        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: queryText,
                top_k: parseInt(document.getElementById("input-top-k")?.value) || 5
            })
        });

        if (!response.ok) {
            throw new Error(`Máy chủ phản hồi mã lỗi ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop();

            for (const block of lines) {
                const line = block.trim();
                if (!line.startsWith("data: ")) continue;
                const jsonStr = line.substring(6).trim();
                if (!jsonStr) continue;

                try {
                    const evt = JSON.parse(jsonStr);

                    if (evt.type === "meta") {
                        if (evt.matched_diseases && evt.matched_diseases.length > 0) {
                            candContainerEl.style.display = "block";
                            
                            const fusionModeLabel = {
                                "consensus": "⚡ Consensus (Đồng thuận KG & RAG)",
                                "kg_high_conf": "🔒 KG High Confidence (Độ tin cậy KG cao)",
                                "disagree_kg_anchored": "⚖️ Disagree (Neo đồ thị KG)",
                                "retrieval_only": "🔍 Retrieval-Only + BGE Rerank"
                            }[evt.fusion_mode] || (evt.fusion_mode ? `Chế độ: ${evt.fusion_mode}` : "");

                            const symptomsHtml = (evt.symptoms_extracted && evt.symptoms_extracted.length > 0)
                                ? `<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; align-items: center;">
                                     <span style="font-size: 0.78rem; color: var(--gemini-text-muted); font-weight: 500;">Triệu chứng bóc tách:</span>
                                     ${evt.symptoms_extracted.map(s => `<span style="font-size: 0.76rem; padding: 2px 8px; border-radius: 12px; background: rgba(66, 133, 244, 0.15); color: #8ab4f8; border: 1px solid rgba(66, 133, 244, 0.3); font-weight: 500;">${escapeHtml(s)}</span>`).join('')}
                                   </div>`
                                : '';

                            const fusionBadgeHtml = fusionModeLabel
                                ? `<span style="font-size: 0.74rem; padding: 2px 8px; border-radius: 10px; background: rgba(52, 168, 83, 0.15); color: #81c995; border: 1px solid rgba(52, 168, 83, 0.3); font-weight: 500; margin-left: 8px;">${fusionModeLabel}</span>`
                                : '';

                            candContainerEl.innerHTML = `
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                                    <div style="font-size: 0.88rem; font-weight: 600; color: var(--gemini-text-main); display: flex; align-items: center;">
                                        Top Ứng Viên Bệnh Lý (MedKG-HRR v16)
                                        ${fusionBadgeHtml}
                                    </div>
                                </div>
                                ${symptomsHtml}
                                <div class="candidates-cards-grid">
                                    ${evt.matched_diseases.map(item => renderCandidateCard(item)).join('')}
                                </div>
                            `;
                            scrollViewport();
                        }
                    } else if (evt.type === "token") {
                        accumulatedText += evt.content;
                        // Khi bắt đầu in câu trả lời: 3 chấm tự động biến mất và text hiện ra không có con trỏ xanh
                        textContentEl.innerHTML = parseMarkdown(accumulatedText);
                        scrollViewport();
                    } else if (evt.type === "done") {
                        textContentEl.innerHTML = parseMarkdown(accumulatedText);
                        renderActionBar(actionBarEl, msgId);
                        scrollViewport();
                    } else if (evt.type === "error") {
                        renderErrorRow(evt.content || "Lỗi xử lý chẩn đoán.");
                    }
                } catch (pe) {
                    console.error("Lỗi parse SSE:", pe, jsonStr);
                }
            }
        }

        textContentEl.innerHTML = parseMarkdown(accumulatedText);
        renderActionBar(actionBarEl, msgId);
    } catch (err) {
        renderErrorRow("Không thể kết nối máy chủ hoặc gián đoạn stream: " + err.message);
    } finally {
        state.isGenerating = false;
        scrollViewport();
    }
}

function renderActionBar(actionBarEl, msgId) {
    if (!actionBarEl) return;
    actionBarEl.style.display = "flex";
    actionBarEl.innerHTML = `
        <button class="action-btn-circle" onclick="speakResponse('${msgId}')" title="Nghe câu trả lời">
            <i class="fa-solid fa-volume-high"></i>
        </button>
        <button class="action-btn-circle" onclick="copyResponse('${msgId}')" title="Sao chép">
            <i class="fa-regular fa-copy"></i>
        </button>
        <button class="action-btn-circle" onclick="showToast('Cảm ơn phản hồi của bạn!')" title="Hữu ích">
            <i class="fa-regular fa-thumbs-up"></i>
        </button>
        <button class="action-btn-circle" onclick="showToast('Đã ghi nhận phản hồi')" title="Chưa tốt">
            <i class="fa-regular fa-thumbs-down"></i>
        </button>
        <button class="action-btn-circle" onclick="showToast('Đã tạo liên kết chia sẻ')" title="Chia sẻ">
            <i class="fa-solid fa-share-nodes"></i>
        </button>
    `;
}

function renderCandidateCard(item) {
    const simPct = item.similarity_pct || 80;
    const cardId = 'drawer-' + Math.random().toString(36).substr(2, 9);
    const kgCardId = 'kg-drawer-' + Math.random().toString(36).substr(2, 9);

    const sourceBadge = item.source ? `
        <span style="font-size: 0.7rem; padding: 1px 6px; border-radius: 8px; background: rgba(255, 255, 255, 0.08); color: var(--gemini-text-muted); border: 1px solid rgba(255, 255, 255, 0.12);">${escapeHtml(item.source)}</span>
    ` : '';

    return `
        <div class="gemini-candidate-pill-card">
            <div>
                <div class="cand-top-row">
                    <span class="cand-name">${escapeHtml(item.disease)}</span>
                    <div style="display: flex; align-items: center; gap: 4px;">
                        ${sourceBadge}
                        ${item.icd10 ? `<span class="cand-icd">${item.icd10}</span>` : ''}
                    </div>
                </div>
                <div class="cand-meta">${escapeHtml(item.specialty || "Đa khoa")} • Khớp ${simPct}%</div>
            </div>

            ${item.kg_evidence && item.kg_evidence.length > 0 ? `
                <div style="margin-top: 6px;">
                    <div class="cand-drawer-toggle" onclick="toggleDrawer('${kgCardId}')" style="color: #81c995;">
                        <span><i class="fa-solid fa-diagram-project" style="margin-right: 4px;"></i> Tri thức Đồ thị KG</span> <i class="fa-solid fa-chevron-down"></i>
                    </div>
                    <div class="cand-drawer-content" id="${kgCardId}">
                        <strong>Triệu chứng liên kết trong đồ thị:</strong><br/>
                        ${item.kg_evidence.map(s => `• ${escapeHtml(s)}`).join('<br/>')}
                    </div>
                </div>
            ` : ''}

            ${item.chunk_evidence ? `
                <div style="margin-top: 4px;">
                    <div class="cand-drawer-toggle" onclick="toggleDrawer('${cardId}')">
                        <span>Bằng chứng trích xuất</span> <i class="fa-solid fa-chevron-down"></i>
                    </div>
                    <div class="cand-drawer-content" id="${cardId}">
                        ${escapeHtml(item.chunk_evidence)}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

window.toggleDrawer = function (id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("open");
};

function renderErrorRow(err) {
    const stream = document.getElementById("messages-stream");
    const row = document.createElement("div");
    row.className = "gemini-msg-row ai-row";
    row.innerHTML = `
        <div class="gemini-ai-body">
            <div style="color: #f28b82; padding: 12px; background: rgba(217, 101, 112, 0.1); border-radius: 12px;">
                <i class="fa-solid fa-circle-exclamation"></i> ${escapeHtml(err)}
            </div>
        </div>
    `;
    stream.appendChild(row);
    scrollViewport();
}

function scrollViewport() {
    const vp = document.getElementById("chat-viewport");
    if (vp) vp.scrollTop = vp.scrollHeight;
}

// ============================================================
// MARKDOWN PARSER
// ============================================================
function parseMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    html = html.replace(/\|(.+)\|/g, (match, content) => {
        const cells = content.split('|').map(c => c.trim());
        if (cells.every(c => c.match(/^:?-+:?$/))) {
            return '<tr class="divider"></tr>';
        }
        const isHeader = match.includes('---');
        const tag = isHeader ? 'th' : 'td';
        return `<tr>${cells.map(c => `<${tag}>${c}</${tag}>`).join('')}</tr>`;
    });
    html = html.replace(/((?:<tr>.*?<\/tr>\s*)+)/g, '<table>$1</table>');
    html = html.replace(/<tr class="divider"><\/tr>/g, '');

    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h3>$1</h3>');

    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    html = html.replace(/^\s*\*\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/((?:<li>.*?<\/li>\s*)+)/g, '<ul>$1</ul>');

    html = html.replace(/\n\n/g, '<p></p>');
    html = html.replace(/\n/g, '<br>');

    return html;
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// ============================================================
// ACTIONS & MODALS
// ============================================================
window.copyResponse = function (msgId) {
    const el = document.getElementById(msgId);
    if (!el) return;
    const txt = el.querySelector(".text-content")?.innerText || "";
    navigator.clipboard.writeText(txt).then(() => {
        showToast("Đã sao chép vào bộ nhớ tạm");
    });
};

window.speakResponse = function (msgId) {
    const el = document.getElementById(msgId);
    if (!el) return;
    const txt = el.querySelector(".text-content")?.innerText || "";

    if (!('speechSynthesis' in window)) {
        showToast("Trình duyệt không hỗ trợ phát âm");
        return;
    }
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        showToast("Đã dừng đọc");
        return;
    }

    // Làm sạch Markdown, bảng biểu, ký tự đặc biệt
    const clean = txt
        .replace(/[\*\#\_\`\~]/g, '')
        .replace(/\|/g, ', ')
        .replace(/https?:\/\/\S+/g, '')
        .replace(/\n+/g, '. ')
        .replace(/\s+/g, ' ')
        .trim();

    if (!clean) return;

    const utter = new SpeechSynthesisUtterance(clean.substring(0, 1800));
    utter.lang = 'vi-VN';
    utter.rate = 1.0;
    utter.pitch = 1.0;

    // Lấy danh sách voice và ưu tiên giọng Tiếng Việt
    const voices = window.speechSynthesis.getVoices();
    const vietnameseVoice = voices.find(v =>
        v.lang === 'vi-VN' ||
        v.lang === 'vi_VN' ||
        v.lang.toLowerCase().startsWith('vi') ||
        v.name.toLowerCase().includes('vietnam') ||
        v.name.toLowerCase().includes('tiếng việt') ||
        v.name.toLowerCase().includes('hoaimy') ||
        v.name.toLowerCase().includes('namminh')
    );

    if (vietnameseVoice) {
        utter.voice = vietnameseVoice;
    }

    utter.onstart = () => {
        showToast("Đang phát âm thanh tiếng Việt...");
    };

    utter.onend = () => {
        // Đọc xong tự nhiên
    };

    utter.onerror = (e) => {
        console.warn("Speech error:", e);
    };

    window.speechSynthesis.speak(utter);
};

window.openSettingsModal = function () {
    document.getElementById("modal-settings").classList.add("open");
    loadSystemConfig();
};

window.closeSettingsModal = function () {
    document.getElementById("modal-settings").classList.remove("open");
    document.getElementById("test-llm-result").innerHTML = "";
};

function toggleKeyVisibility() {
    const input = document.getElementById("input-gemini-key");
    const icon = document.querySelector("#btn-toggle-key-visibility i");
    if (input.type === "password") {
        input.type = "text";
        icon.className = "fa-regular fa-eye-slash";
    } else {
        input.type = "password";
        icon.className = "fa-regular fa-eye";
    }
}

async function handleTestLLMConnection() {
    const indicator = document.getElementById("test-llm-result");
    const key = document.getElementById("input-gemini-key").value.trim();
    const model = document.getElementById("select-gemini-model").value;

    indicator.innerHTML = '<span style="color: #4285f4;"><i class="fa-solid fa-spinner fa-spin"></i> Đang kiểm tra...</span>';

    try {
        const res = await fetch("/api/test-llm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key || undefined, model: model })
        });
        const data = await res.json();
        if (data.success) {
            indicator.innerHTML = '<span style="color: #34a853;"><i class="fa-solid fa-circle-check"></i> Kết nối thành công!</span>';
        } else {
            indicator.innerHTML = `<span style="color: #d96570;">Lỗi: ${escapeHtml(data.message)}</span>`;
        }
    } catch (e) {
        indicator.innerHTML = `<span style="color: #d96570;">Lỗi: ${e.message}</span>`;
    }
}

async function handleSaveSettings(e) {
    e.preventDefault();
    const key = document.getElementById("input-gemini-key").value.trim();
    const model = document.getElementById("select-gemini-model").value;
    const temp = parseFloat(document.getElementById("input-temperature").value);
    const topK = parseInt(document.getElementById("input-top-k").value);

    try {
        const res = await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                gemini_api_key: key || undefined,
                gemini_model: model,
                temperature: temp,
                top_k: topK
            })
        });
        if (res.ok) {
            showToast("Đã lưu cài đặt");
            closeSettingsModal();
            loadSystemStatus();
        }
    } catch (e) {
        showToast("Lỗi lưu cấu hình: " + e.message);
    }
}

// Explorer Modal
window.openExplorerModal = async function () {
    document.getElementById("modal-explorer").classList.add("open");
    const container = document.getElementById("explorer-results-list");

    if (state.explorerData.length === 0) {
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: #8e918f;"><i class="fa-solid fa-spinner fa-spin"></i> Đang tải danh mục 1.109 bệnh...</div>';
        try {
            const res = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: "bệnh sốt ho viêm đau", top_k: 25 })
            });
            const data = await res.json();
            state.explorerData = data.results || [];
        } catch (e) {
            container.innerHTML = '<div style="color: #d96570;">Lỗi tải dữ liệu</div>';
            return;
        }
    }

    renderExplorer(state.explorerData);
};

window.closeExplorerModal = function () {
    document.getElementById("modal-explorer").classList.remove("open");
};

function renderExplorer(items) {
    const container = document.getElementById("explorer-results-list");
    if (!items || items.length === 0) {
        container.innerHTML = '<div style="padding: 20px; color: #8e918f; text-align: center;">Không tìm thấy bệnh phù hợp.</div>';
        return;
    }
    container.innerHTML = items.map(it => `
        <div class="explorer-item-row" onclick="selectExplorerItem('${escapeHtml(it.disease)}')">
            <div class="exp-row-header">
                <span class="exp-disease">${escapeHtml(it.disease)}</span>
                <span class="cand-icd">${escapeHtml(it.icd10 || "ICD-10")}</span>
            </div>
            <div class="exp-body">${escapeHtml(it.specialty || "Đa khoa")} • ${escapeHtml(it.symptom || "Đang cập nhật")}</div>
        </div>
    `).join('');
}

async function filterExplorer(q) {
    if (!q.trim()) {
        renderExplorer(state.explorerData);
        return;
    }
    try {
        const res = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: q, top_k: 25 })
        });
        const data = await res.json();
        renderExplorer(data.results || []);
    } catch (e) { }
}

window.selectExplorerItem = function (name) {
    closeExplorerModal();
    const input = document.getElementById("user-input");
    input.value = `Tìm hiểu thông tin và triệu chứng của bệnh: ${name}`;
    document.getElementById("btn-send").style.display = "flex";
    document.getElementById("btn-capsule-mic").style.display = "none";
    input.focus();
};

function showToast(msg) {
    const c = document.getElementById("toast-container");
    const t = document.createElement("div");
    t.className = "toast";
    t.innerHTML = `<i class="fa-solid fa-check"></i> <span>${escapeHtml(msg)}</span>`;
    c.appendChild(t);
    setTimeout(() => {
        t.style.opacity = "0";
        setTimeout(() => t.remove(), 300);
    }, 2800);
}
