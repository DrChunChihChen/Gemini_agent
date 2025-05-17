# Gemini_agent
📘 AI CSV 分析助手 v3.1 - 使用說明（繁體中文）

🔍 專案簡介

本應用程式為一款基於 Streamlit 的互動式資料分析工具，結合 Google Gemini 2.0 Flash / Flash Lite 模型，讓使用者可：

上傳 CSV 檔案

根據資料產生多部門主管觀點（CEO、CFO、CTO、COO、CMO）

彙整 5 個可執行的分析建議

自動產生對應的 Python 程式碼並執行

產出報表或繪圖

最後由另一個 LLM 擔任 "評審" 提出批判建議

🧠 使用到的 AI 模型

模型角色

模型名稱

說明

分析執行者

gemini-2.0-flash-lite

快速產生分析視角、程式碼、報告

分析評審者

gemini-2.0-flash

提供更完整、深入的批判與建議

如未設定 API 金鑰，則將以 PlaceholderLLM 模擬模型行為。

🛠️ 安裝方式

pip install streamlit pandas matplotlib seaborn langchain-google-genai langchain-core

📂 檔案結構

simplified_data_agent_streamlit_v3_enhanced.py  # 主程式碼
/temp_data_simplified_agent/                    # 執行後自動建立，儲存暫存圖表/報告/結果

🚀 如何使用

啟動應用：

streamlit run simplified_data_agent_streamlit_v3_enhanced.py

設定 Google Gemini API 金鑰：

建議設定於 st.secrets 或系統環境變數 LLM_API_KEY

於介面左側：

選擇 Worker 與 Judge 模型（例如 gemini-2.0-flash-lite / flash）

上傳你的 CSV 檔案

程式將：

產生多部門主管觀點

整合分析策略建議

根據你提出的分析問題產生程式碼並執行

顯示表格、圖片或文字結果

提供自動產生報表

按一下按鈕讓 Judge 模型進行批判分析

🧱 技術架構

📦 Streamlit：建構 UI + 狀態管理

🧠 LangChain + Gemini API：提示設計與記憶

📈 Matplotlib / Seaborn：產生圖表

📊 Pandas：資料整理與分析

⚙️ Exec()：執行 AI 產生的程式碼（⚠️ 僅供開發/教育用途）

🔐 安全提醒

本應用會使用 exec() 執行 LLM 所產生的 Python 程式碼。

僅供開發與展示用途，勿部署於公開或不可信環境。

💡 特色亮點

支援部門角色導向分析

即時產出 Python 程式與執行結果

支援圖表、表格、報告與批判全流程

模型自由切換、記憶上下文脈絡

🤝 貢獻方式

歡迎 pull request 或 issue：優化 prompt、強化安全性、模組化功能皆可。

🧪 測試建議

測試資料建議含有：分類欄位、數值欄位、時間欄位

建議逐步嘗試：

上傳資料 → 看觀點

輸入分析問題 → 產程式碼與結果

點報告與評審按鈕 → 檢查產物內容
