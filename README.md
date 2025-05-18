# Gemini_agent (繁體中文)

本專案是一個互動式CSV數據分析工具，採用Streamlit框架，並整合Google Gemini AI模型，旨在協助使用者快速理解數據、產生分析策略、執行分析並獲得報告與評價。

## 🌟 特色亮點

* **CDO引導的分析流程**：由AI扮演的數據長(CDO)首先對數據進行初步描述與品質評估，接著由其他AI部門主管(CEO, CFO, CTO, COO, CMO)提供觀點，最後CDO整合所有資訊提出5項可執行的分析策略（聚焦於視覺化、表格與描述性方法）。
* **角色導向分析**：模擬多部門主管視角，提供多元化的分析切入點。
* **自動化程式碼生成與執行**：根據使用者查詢或AI建議的策略，自動生成Python分析程式碼並執行。
* **多樣化產出**：支援圖表、表格、文字報告的產出。
* **AI評審機制**：由獨立的AI評審模型對分析結果進行批判性評估。
* **完整報告匯出**：可將分析目標、CDO數據品質報告、圖表、相關數據、分析報告及最終評論匯出為單一PDF檔案。
* **模型彈性切換**：使用者可自由選擇用於分析執行和評審的AI模型。
* **上下文記憶**：在對話過程中保持記憶，使分析更連貫。

## 🎬 操作演示

<a href="https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_PLACEHOLDER" target="_blank">
 <img src="https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_PLACEHOLDER/0.jpg" alt="操作演示影片" style="width:560px;height:315px;" />
</a>

[點此觀看操作演示影片](https://www.youtube.com/watch?v=o7GoR2CViss) ## 🧠 使用到的 AI 模型

| 模型角色     | 模型名稱              | 說明                                         |
| :----------- | :-------------------- | :------------------------------------------- |
| CDO/分析執行者 | gemini-2.0-flash-lite | 進行初步數據描述、產生分析視角、程式碼、報告 |
| 分析評審者   | gemini-2.0-flash      | 提供更完整、深入的批判與建議                 |

*註：如未設定 API 金鑰，則將以 `PlaceholderLLM` 模擬模型行為。*

## 🛠️ 安裝方式

```bash
pip install streamlit pandas matplotlib seaborn langchain-google-genai langchain-core reportlab
