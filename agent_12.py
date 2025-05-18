# simplified_data_agent_streamlit_v3_enhanced.py

import streamlit as st
import pandas as pd
import os
import io  # To read CSV content as string
import json  # For data summary
import datetime  # For timestamping saved files
import matplotlib.pyplot  # Explicit import for placeholder and execution scope
import seaborn  # Explicit import for placeholder and execution scope
# import numpy as np # Only needed if PlaceholderLLM directly uses it;
# actual generated code should import it.

# --- PDF Export ---
# pip install reportlab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# --- Langchain/LLM Components ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# --- Configuration ---
LLM_API_KEY = os.environ.get("LLM_API_KEY", "LLM_API_KEY")
TEMP_DATA_STORAGE = "temp_data_simplified_agent/"
os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)

AVAILABLE_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
DEFAULT_WORKER_MODEL = "gemini-2.0-flash-lite"
DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"

if LLM_API_KEY == "YOUR_API_KEY_HERE":  # This should ideally match the default in os.environ.get
    st.error(
        "Please set your LLM_API_KEY (e.g., GOOGLE_API_KEY) environment variable or Streamlit secret for full functionality.")


# --- LLM Initialization & Placeholder ---
class PlaceholderLLM:
    """Simulates LLM responses for when an API key is not available."""

    def __init__(self, model_name="placeholder_model"):
        self.model_name = model_name
        st.warning(f"Using PlaceholderLLM for {self.model_name} as API key is not set or invalid.")

    def invoke(self, prompt_input):
        prompt_str_content = str(prompt_input.to_string() if hasattr(prompt_input, 'to_string') else prompt_input)

        if "CDO, your first task is to provide an initial description of the dataset" in prompt_str_content:
            data_summary_json = {}
            try:
                summary_marker = "Data Summary (for context):"
                if summary_marker in prompt_str_content:
                    json_str_part = \
                    prompt_str_content.split(summary_marker)[1].split("\n\nDetailed Initial Description by CDO:")[
                        0].strip()
                    data_summary_json = json.loads(json_str_part)
            except Exception:
                pass

            cols = data_summary_json.get("columns", ["N/A"])
            num_rows = data_summary_json.get("num_rows", "N/A")
            num_cols = data_summary_json.get("num_columns", "N/A")
            dtypes_str = "\n".join(
                [f"- {col}: {data_summary_json.get('dtypes', {}).get(col, 'Unknown')}" for col in cols])

            return {"text": f"""
*Placeholder CDO Initial Data Description ({self.model_name}):*

**1. Dataset Overview (Simulated df.info()):**
   - Rows: {num_rows}, Columns: {num_cols}
   - Column Data Types:
{dtypes_str}
   - Potential Memory Usage: (Placeholder value) MB

**2. Inferred Meaning of Variables (Example):**
   - `ORDERNUMBER`: Unique identifier for each order.
   - `QUANTITYORDERED`: Number of units for a product in an order.
   *(This is a generic interpretation; actual meanings depend on the dataset.)*

**3. Initial Data Quality Assessment (Example):**
   - **Missing Values:** (Placeholder - e.g., "Column 'ADDRESSLINE2' has 80% missing values.")
   - **Overall:** The dataset seems reasonably structured.
"""}

        elif "panel of expert department heads, including the CDO" in prompt_str_content:
            return {"text": """
*Placeholder Departmental Perspectives (after CDO's initial report, via {model_name}):*

**CEO:** Focus on revenue trends.
**CFO:** Assess regional profitability.
**CDO (Highlighting for VPs):** Consider missing values.
""".format(model_name=self.model_name)}

        elif "You are the Chief Data Officer (CDO) of the company." in prompt_str_content and "synthesize these diverse perspectives" in prompt_str_content:
            return {"text": """
*Placeholder Final Analysis Strategy (Synthesized by CDO, via {model_name}):*

1.  **Visualize Core Sales Trends:** Line plot of 'SALES' over 'ORDERDATE'.
2.  **Tabulate Product Line Performance:** Table of 'SALES', 'PRICEEACH', 'QUANTITYORDERED' by 'PRODUCTLINE'.
3.  **Descriptive Summary of Order Status:** Table count by 'STATUS'.
4.  **Data Quality Table for Key Columns:** Table of missing value % for key columns.
5.  **Visualize Sales by Country:** Bar chart of 'SALES' by 'COUNTRY'.
""".format(model_name=self.model_name)}
        elif "Python code:" in prompt_str_content and "User Query:" in prompt_str_content:  # Code generation
            user_query_segment = prompt_str_content.split("User Query:")[1].split("\n")[0].lower()
            # Default fallback message if no specific logic matches
            fallback_script = """
# Standard library imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Ensure numpy is available if used by generated code

# Initialize analysis_result with a default message.
# This will be overwritten if the script successfully assigns a more specific result.
analysis_result = "Analysis logic executed. If you expected a specific output, please check the generated script."
plot_data_df = None # Initialize plot_data_df

# --- AI Generated Code Start ---
# Placeholder: The AI would generate its specific analysis logic here.
# For example, if the query was 'show head':
# analysis_result = df.head()
# plot_data_df = df.head().copy()
#
# Or for a plot:
# fig, ax = plt.subplots()
# if not df.empty and 'SALES' in df.columns:
#   ax.hist(df['SALES'])
#   plot_data_df = df[['SALES']].copy()
#   plot_path = f"{TEMP_DATA_STORAGE}sales_histogram.png"
#   plt.savefig(plot_path)
#   plt.close(fig)
#   analysis_result = plot_path
# else:
#   analysis_result = "Could not generate plot; 'SALES' column missing or df empty."
# --- AI Generated Code End ---

# Fallback: Ensure analysis_result is explicitly set if not by AI's code.
# This is a safety net. The AI should ideally set it.
if 'analysis_result' not in locals() or analysis_result == "Analysis logic executed. If you expected a specific output, please check the generated script.":
    if isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set by the AI's main logic. Displaying df.head() as a default."
        plot_data_df = df.head().copy() # Provide some default data for reporting
    else:
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set, and no DataFrame was available."
"""
            if "average sales" in user_query_segment:
                return {"text": "analysis_result = df['sales'].mean()\nplot_data_df = None"}
            elif "plot" in user_query_segment or "visualize" in user_query_segment:
                generated_plot_code = f"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots()
if not df.empty and len(df.columns) > 0:
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        ax.hist(df[numeric_cols[0]])
        plot_data_df = df[[numeric_cols[0]]].copy()
        analysis_result = '{TEMP_DATA_STORAGE}placeholder_plot.png' # Path to plot
        plt.savefig(analysis_result)
        plt.close(fig)
    else: # Non-numeric data, try a bar plot of value counts of the first column
        if not df.empty:
            try:
                counts = df.iloc[:, 0].value_counts().head(10) # Top 10 for readability
                counts.plot(kind='bar', ax=ax)
                plt.title(f'Value Counts for {df.columns[0]}')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_data_df = counts.reset_index()
                plot_data_df.columns = [df.columns[0], 'count']
                analysis_result = '{TEMP_DATA_STORAGE}placeholder_plot.png' # Path to plot
                plt.savefig(analysis_result)
                plt.close(fig)
            except Exception as e:
                ax.text(0.5, 0.5, 'Could not generate fallback plot.', ha='center', va='center')
                plot_data_df = pd.DataFrame()
                analysis_result = "Failed to generate fallback plot: " + str(e)
                plt.close(fig) # Close the figure even if plotting failed
        else:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
            plot_data_df = pd.DataFrame()
            analysis_result = "No data to plot" # Set analysis_result message
            plt.close(fig) # Close the figure
else:
    ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
    plot_data_df = pd.DataFrame()
    analysis_result = "DataFrame is empty, cannot plot." # Set analysis_result message
    plt.close(fig) # Close the figure if df is empty initially
"""
                return {"text": generated_plot_code}
            elif "table" in user_query_segment or "summarize" in user_query_segment:
                return {
                    "text": "analysis_result = df.describe()\nplot_data_df = df.describe().reset_index()"}
            else:  # Default for other queries, or if the LLM doesn't follow instructions
                return {"text": fallback_script.replace("{TEMP_DATA_STORAGE}", TEMP_DATA_STORAGE)}

        elif "Generate a textual report" in prompt_str_content:
            return {
                "text": f"### Placeholder Report ({self.model_name})\nThis is a placeholder report based on the CDO's focused analysis strategy."}
        elif "Critique the following analysis artifacts" in prompt_str_content:
            return {"text": f"""
### Placeholder Critique ({self.model_name})
**Overall Assessment:** Placeholder.
**Python Code:** Placeholder.
**Data:** Placeholder.
**Report:** Placeholder.
**Suggestions for Worker AI:** Placeholder.
"""}
        else:
            return {
                "text": f"Placeholder response from {self.model_name} for unrecognized prompt: {prompt_str_content[:200]}..."}


def get_llm_instance(model_name: str):
    if not model_name:
        st.error("No model name provided for LLM initialization.")
        return None
    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}
    if model_name not in st.session_state.llm_cache:
        if not LLM_API_KEY or LLM_API_KEY == "LLM_API_KEY":
            st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
        else:
            try:
                temperature = 0.7 if st.session_state.get("selected_judge_model", "") == model_name else 0.2
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=LLM_API_KEY,
                    temperature=temperature,
                    convert_system_message_to_human=True
                )
                st.session_state.llm_cache[model_name] = llm
            except Exception as e:
                st.error(f"Failed to initialize Gemini LLM ({model_name}): {e}")
                st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
    return st.session_state.llm_cache[model_name]


def load_csv_and_get_summary(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.current_dataframe = df
        st.session_state.data_source_name = uploaded_file.name
        st.session_state.current_analysis_artifacts = {}
        data_summary = {
            "source_name": uploaded_file.name,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values_per_column": df.isnull().sum().to_dict(),
            "descriptive_stats_sample": df.describe(include='all').to_json() if not df.empty else "N/A",
            "preview_head": df.head().to_dict(orient='records'),
            "preview_tail": df.tail().to_dict(orient='records')
        }
        st.session_state.data_summary = data_summary
        return True
    except Exception as e:
        st.error(f"Error loading CSV or generating summary: {e}")
        st.session_state.current_dataframe = None
        st.session_state.data_summary = None
        return False


class LocalCodeExecutionEngine:
    def execute_code(self, code_string, df_input):
        if df_input is None:
            return {"type": "error", "message": "No data loaded to execute code on."}
        exec_globals = globals().copy()
        exec_globals['plt'] = matplotlib.pyplot
        exec_globals['sns'] = seaborn
        exec_globals['pd'] = pd
        # Add numpy to the execution scope as it's often used and imported in generated code
        import numpy as np
        exec_globals['np'] = np

        local_scope = {'df': df_input.copy(), 'pd': pd, 'plt': matplotlib.pyplot, 'sns': seaborn, 'np': np}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_df_saved_path = None

        # Initialize analysis_result with the default message *before* exec
        # This ensures it exists in local_scope if the exec'd code doesn't set it
        # or if the exec'd code has an early exit or error before setting it.
        default_analysis_result_message = "Code executed, but 'analysis_result' was not explicitly set by the script."
        local_scope['analysis_result'] = default_analysis_result_message
        local_scope['plot_data_df'] = None  # Ensure plot_data_df is also initialized

        try:
            exec(code_string, exec_globals, local_scope)

            # Retrieve results AFTER exec. If the script modified them, we get the new values.
            # If not, we get the pre-initialized default values.
            analysis_result = local_scope.get('analysis_result')
            plot_data_df = local_scope.get('plot_data_df')

            # Check if analysis_result is still the default message.
            # This means the AI's script ran but didn't assign to analysis_result.
            # This check is now more of a logging/debugging aid, as the prompt
            # guides the AI to include its own fallback.
            if analysis_result == default_analysis_result_message:
                st.warning(
                    "The executed script did not explicitly set the 'analysis_result' variable. The output might be incomplete or not as expected.")

            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                return {"type": "error", "message": analysis_result}
            if isinstance(analysis_result, str) and analysis_result.endswith((".png", ".jpg", ".jpeg", ".svg")):
                relative_plot_path = os.path.basename(analysis_result)
                potential_paths = [os.path.join(TEMP_DATA_STORAGE, relative_plot_path), analysis_result,
                                   relative_plot_path]
                final_plot_path = next((p for p in potential_paths if os.path.exists(p)), None)
                if not final_plot_path:
                    intended_path_msg = analysis_result if TEMP_DATA_STORAGE not in analysis_result else os.path.join(
                        TEMP_DATA_STORAGE, relative_plot_path)
                    return {"type": "error",
                            "message": f"Plot file '{relative_plot_path}' (intended: '{intended_path_msg}') not found. Checked: {potential_paths}"}
                if isinstance(plot_data_df, pd.DataFrame) and not plot_data_df.empty:
                    plot_data_filename = f"plot_data_for_{os.path.splitext(relative_plot_path)[0]}_{timestamp}.csv"
                    plot_data_df_saved_path = os.path.join(TEMP_DATA_STORAGE, plot_data_filename)
                    plot_data_df.to_csv(plot_data_df_saved_path, index=False)
                    st.info(f"Plot-specific data saved to: {plot_data_df_saved_path}")
                elif plot_data_df is not None:  # It was set, but not a valid DF
                    st.warning(
                        "`plot_data_df` was set by the script but is not a valid or non-empty DataFrame. Not saving associated data for the plot.")
                return {"type": "plot", "plot_path": final_plot_path, "data_path": plot_data_df_saved_path}
            elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                analysis_result = analysis_result.to_frame() if isinstance(analysis_result,
                                                                           pd.Series) else analysis_result
                if analysis_result.empty: return {"type": "text", "value": "The analysis resulted in an empty table."}
                saved_csv_path = os.path.join(TEMP_DATA_STORAGE, f"table_result_{timestamp}.csv")
                analysis_result.to_csv(saved_csv_path, index=False)
                return {"type": "table", "data_path": saved_csv_path}
            else:  # Covers text, numbers, or the default message if not overwritten
                return {"type": "text", "value": str(analysis_result)}
        except Exception as e:
            import traceback
            # If an error occurs during exec, analysis_result might not have been set by the script.
            # The default from before exec might be used, or it might be what the script set before erroring.
            # It's also possible the error itself is the most useful result.
            error_message_for_user = f"Error during code execution: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            # Ensure analysis_result reflects the error if it's still the default or None
            current_analysis_res = local_scope.get('analysis_result', default_analysis_result_message)
            if current_analysis_res == default_analysis_result_message or current_analysis_res is None:
                local_scope['analysis_result'] = f"Execution Error: {str(e)}"

            return {"type": "error", "message": error_message_for_user,
                    "final_analysis_result_value": local_scope['analysis_result']}


code_executor = LocalCodeExecutionEngine()


# --- PDF Export Function ---
def export_analysis_to_pdf(artifacts, output_filename="analysis_report.pdf"):
    """Exports the analysis artifacts to a PDF file."""
    pdf_path = os.path.join(TEMP_DATA_STORAGE, output_filename)
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Comprehensive Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # 1. Analysis Goal
    story.append(Paragraph("1. Analysis Goal (User Query)", styles['h2']))
    analysis_goal = artifacts.get("original_user_query", "Not specified.")
    story.append(Paragraph(analysis_goal, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # 2. CDO's Initial Data Description & Quality Assessment
    story.append(Paragraph("2. CDO's Initial Data Description & Quality Assessment", styles['h2']))
    cdo_report_text = st.session_state.get("cdo_initial_report_text", "CDO initial report not available.")
    # Clean up markdown-like bolding for PDF
    cdo_report_text_cleaned = cdo_report_text.replace("**", "")
    for para_text in cdo_report_text_cleaned.split('\n'):
        if para_text.strip().startswith("- "):  # Handle bullet points
            story.append(Paragraph(para_text, styles['Bullet'], bulletText='-'))
        elif para_text.strip():
            story.append(Paragraph(para_text, styles['Normal']))
        else:
            story.append(Spacer(1, 0.1 * inch))  # Add small spacer for empty lines
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())

    # 3. Generated Plot
    story.append(Paragraph("3. Generated Plot", styles['h2']))
    plot_image_path = artifacts.get("plot_image_path")
    if plot_image_path and os.path.exists(plot_image_path):
        try:
            img = Image(plot_image_path, width=6 * inch, height=4 * inch)
            img.hAlign = 'CENTER'
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Error embedding plot: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Plot image not available or path incorrect.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # 4. Plot Data (CSV Content) or Executed Data Table
    story.append(Paragraph("4. Plot Data (or Executed Data Table)", styles['h2']))
    plot_data_csv_path = artifacts.get("executed_data_path")
    if plot_data_csv_path and os.path.exists(plot_data_csv_path) and plot_data_csv_path.endswith(".csv"):
        try:
            df_plot = pd.read_csv(plot_data_csv_path)
            data_for_table = [df_plot.columns.to_list()] + df_plot.values.tolist()
            if len(data_for_table) > 1:
                max_rows_in_pdf = 30
                if len(data_for_table) > max_rows_in_pdf:
                    data_for_table = data_for_table[:max_rows_in_pdf]
                    story.append(
                        Paragraph(f"(Showing first {max_rows_in_pdf - 1} data rows of the CSV)", styles['Italic']))

                table = Table(data_for_table, repeatRows=1)
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                table.setStyle(table_style)
                story.append(table)
            else:
                story.append(Paragraph("CSV file is empty or contains only headers.", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading or displaying CSV data: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Plot data CSV not available or path incorrect.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())

    # 5. Generated Textual Report (from specific analysis)
    story.append(Paragraph("5. Generated Textual Report (Specific Analysis)", styles['h2']))
    report_text_path = artifacts.get("generated_report_path")
    if report_text_path and os.path.exists(report_text_path):
        try:
            with open(report_text_path, 'r', encoding='utf-8') as f:
                report_text_content = f.read()
            # Clean up markdown-like bolding for PDF
            report_text_content_cleaned = report_text_content.replace("**", "")
            for para_text in report_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading report text file: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Generated report text file not available.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # 6. Analysis Critique
    story.append(Paragraph("6. Analysis Critique", styles['h2']))
    critique_text_path = artifacts.get("generated_critique_path")
    if critique_text_path and os.path.exists(critique_text_path):
        try:
            with open(critique_text_path, 'r', encoding='utf-8') as f:
                critique_text_content = f.read()
            # Clean up markdown-like bolding for PDF
            critique_text_content_cleaned = critique_text_content.replace("**", "")
            for para_text in critique_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading critique text file: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Critique text file not available.", styles['Normal']))

    try:
        doc.build(story)
        return pdf_path
    except Exception as e:
        st.error(f"Failed to build PDF: {e}")
        return None


st.set_page_config(page_title="AI CSV Analyst v3.1 (CDO Workflow)", layout="wide")
st.title("🤖 AI CSV Analyst v3.1 (CDO Workflow)")
st.caption(
    "Upload CSV, CDO describes data, VPs discuss, CDO synthesizes strategy, then Worker Model codes & Judge critiques.")

if "messages" not in st.session_state: st.session_state.messages = [
    {"role": "assistant", "content": "Hello! Select models, upload CSV to start CDO workflow."}]
if "current_dataframe" not in st.session_state: st.session_state.current_dataframe = None
if "data_summary" not in st.session_state: st.session_state.data_summary = None
if "data_source_name" not in st.session_state: st.session_state.data_source_name = None
if "current_analysis_artifacts" not in st.session_state: st.session_state.current_analysis_artifacts = {}
if "selected_worker_model" not in st.session_state: st.session_state.selected_worker_model = DEFAULT_WORKER_MODEL
if "selected_judge_model" not in st.session_state: st.session_state.selected_judge_model = DEFAULT_JUDGE_MODEL
if "lc_memory" not in st.session_state: st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history",
                                                                                              return_messages=False,
                                                                                              input_key="user_query")
if "cdo_initial_report_text" not in st.session_state: st.session_state.cdo_initial_report_text = None

# --- Prompt Templates ---
cdo_initial_data_description_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history"],
    template="""You are the Chief Data Officer (CDO). A user has uploaded a CSV file.
Data Summary (for context):
{data_summary}
CDO, your first task is to provide an initial description of the dataset. This should include:
1.  A brief overview similar to `df.info()` (column names, non-null counts, dtypes).
2.  Your inferred meaning or common interpretation for each variable/column.
3.  A preliminary assessment of data quality (e.g., obvious missing data patterns, potential outliers you notice from the summary, data type consistency).
This description will be shared with the other department heads (CEO, CFO, CTO, COO, CMO) before they provide their perspectives.
Conversation History (for context, if any):
{chat_history}
Detailed Initial Description by CDO:"""
)
individual_perspectives_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report"],
    template="""You are a panel of expert department heads, including the CDO.
A user has uploaded a CSV file, and the Chief Data Officer (CDO) has provided an initial data description and quality assessment.
Data Summary (Original):
{data_summary}
CDO's Initial Data Description & Quality Report:
--- BEGIN CDO REPORT ---
{cdo_initial_report}
--- END CDO REPORT ---
Based on BOTH the original data summary AND the CDO's initial report, provide a detailed perspective from each of the following roles (CEO, CFO, CTO, COO, CMO).
For each role, outline 2-3 specific questions they would now ask, analyses they would want to perform, or observations they would make, considering the CDO's findings.
The CDO should also provide a brief perspective here, perhaps by reiterating 1-2 critical data quality points from their initial report that the other VPs *must* consider, or by highlighting specific data features that are now more apparent.
Structure your response clearly, with each role's perspective under a bolded heading (e.g., **CEO Perspective:**).
* **CEO (首席執行官 - Chief Executive Officer):**
* **CFO (首席財務官 - Chief Financial Officer):**
* **CTO (首席技術官 - Chief Technology Officer):**
* **COO (首席運營官 - Chief Operating Officer):**
* **CMO (首席行銷官 - Chief Marketing Officer):**
* **CDO (首席數據官 - Reiterating Key Points):**
Conversation History (for context, if any):
{chat_history}
Detailed Perspectives from Department Heads (informed by CDO's initial report):"""
)
synthesize_analysis_suggestions_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report", "generated_perspectives_from_others"],
    template="""You are the Chief Data Officer (CDO) of the company.
A user has uploaded a CSV file. You have already performed an initial data description and quality assessment.
Subsequently, the other department heads (CEO, CFO, CTO, COO, CMO) have provided their perspectives based on your initial findings and the data summary.
Original Data Summary:
{data_summary}
Your Initial Data Description & Quality Report:
--- BEGIN YOUR INITIAL CDO REPORT ---
{cdo_initial_report}
--- END YOUR INITIAL CDO REPORT ---
Perspectives from other Department Heads (CEO, CFO, CTO, COO, CMO):
--- BEGIN OTHER PERSPECTIVES ---
{generated_perspectives_from_others}
--- END OTHER PERSPECTIVES ---
Your task is to synthesize all this information (your initial findings AND the other VPs' inputs) into a concise list of **5 distinct and actionable analysis strategy suggestions** for the user.
These suggestions must prioritize analyses that result in clear visualizations (e.g., charts, plots), well-structured tables, or concise descriptive summaries.
This approach is preferred because it makes the analysis results easier to execute locally and interpret broadly.
Present these 5 suggestions as a numbered list. Each suggestion should clearly state the type of analysis (e.g., "Visualize X...", "Create a table for Y...", "Describe Z...").
Conversation History (for context, if any):
{chat_history}
Final 5 Analysis Strategy Suggestions (Synthesized by the CDO, focusing on visualizations, tables, and descriptive methods, incorporating all prior inputs):"""
)

# Updated code_generation_prompt_template
code_generation_prompt_template = PromptTemplate(
    input_variables=["data_summary", "user_query", "chat_history"],
    template="""You are an expert Python data analysis assistant.
Data Summary:
{data_summary}
User Query: "{user_query}"
Previous Conversation (for context):
{chat_history}

Your task is to generate a Python script to perform the requested analysis on a pandas DataFrame named `df`.
**Crucial Instructions for `analysis_result` and `plot_data_df`:**
1.  **`analysis_result` MUST BE SET**: The primary result of your analysis (e.g., a calculated value, a DataFrame, a plot path string, or a descriptive message) MUST be assigned to a variable named `analysis_result`.
2.  **`plot_data_df` for Plots**: If your analysis involves creating a plot:
    a.  Save the plot to a file (e.g., 'analysis_plot.png' or '{TEMP_DATA_STORAGE}analysis_plot.png').
    b.  Set `analysis_result` to the string path of this saved plot file.
    c.  Create a pandas DataFrame named `plot_data_df` containing ONLY the data directly visualized in the chart. If the plot uses the entire `df`, then `plot_data_df = df.copy()`. If no specific data subset is plotted, `plot_data_df` can be an empty DataFrame or `None`.
3.  **`plot_data_df` for Non-Plots**: If the analysis does NOT produce a plot (e.g., it's a table or a single value), `plot_data_df` should generally be set to `None`. However, if `analysis_result` is a DataFrame that you also want to make available for reporting (like a summary table), you can set `plot_data_df = analysis_result.copy()`.
4.  **Default `analysis_result`**: If the user's query is very general (e.g., "explore the data") and no specific plot or table is generated, assign a descriptive string to `analysis_result` (e.g., "Data exploration performed. Key statistics logged or printed.") or assign `df.head()` to `analysis_result`.
5.  **Imports**: Ensure all necessary libraries (`matplotlib.pyplot as plt`, `seaborn as sns`, `pandas as pd`, `numpy as np` if used) are imported within the script.

**Safety Net - Fallback within your generated script:**
Include the following structure in your generated Python code:
```python
# Initialize analysis_result and plot_data_df at the beginning of your script
analysis_result = "Script started, but 'analysis_result' was not yet set by main logic."
plot_data_df = None

# --- Your main analysis code here ---
# (Example: df_summary = df.describe(); analysis_result = df_summary; plot_data_df = df_summary.copy())
# (Example: plt.hist(df['age']); plot_path = 'plot.png'; plt.savefig(plot_path); analysis_result = plot_path; plot_data_df = df[['age']].copy())
# --- End of your main analysis code ---

# Final check: If by the end of your script, analysis_result is still the initial placeholder,
# set it to a more informative default. This is a last resort.
if analysis_result == "Script started, but 'analysis_result' was not yet set by main logic.":
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "Analysis performed. No specific output was explicitly set to 'analysis_result'. Displaying df.head() as a default."
        # plot_data_df = df.head().copy() # Optionally provide data for this default
    else:
        analysis_result = "Analysis performed, but 'analysis_result' was not set and no DataFrame was available."

```
Output only the raw Python code, starting with imports. Do not include any explanations or markdown formatting around the code block itself.

Python code:""".replace("{TEMP_DATA_STORAGE}", TEMP_DATA_STORAGE)
)

report_generation_prompt_template = PromptTemplate(
    input_variables=["table_data_csv", "original_data_summary", "user_query_that_led_to_data", "chat_history"],
    template="""You are an insightful data analyst. Report based on data and context.
Original Data Summary: {original_data_summary}
User Query for this data: "{user_query_that_led_to_data}"
Chat History: {chat_history}
Analysis Result Data (CSV):
```csv
{table_data_csv}
```
**Report Structure:**
* 1. Executive Summary (1-2 sentences): Main conclusion from "Analysis Result Data" for the query.
* 2. Purpose (1 sentence): User's goal for "{user_query_that_led_to_data}".
* 3. Key Observations (Bulleted list, 2-4 points): From "Analysis Result Data", quantified.
* 4. Actionable Insights (1-2 insights): Meaning of observations in context.
* 5. Data Focus & Limitations: Report based *solely* on "Analysis Result Data". If "N/A", insights are general.
**Tone:** Professional, clear, authoritative. Explain simply. Do NOT say "the CSV".
Report:"""
)
judging_prompt_template = PromptTemplate(
    input_variables=["python_code", "data_csv_content", "report_text_content", "original_user_query", "data_summary",
                     "plot_image_path", "plot_info"],
    template="""Expert data science reviewer. Evaluate AI assistant's artifacts.
Original User Query: "{original_user_query}"
Original Data Summary: {data_summary}
--- ARTIFACTS ---
1. Python Code: ```python\n{python_code}\n```
2. Data Produced (CSV or text): ```csv\n{data_csv_content}\n```\n{plot_info}
3. Report: ```text\n{report_text_content}\n```
--- END ARTIFACTS ---
Critique:
1. Code Quality: Correctness, efficiency, readability, best practices, bugs? Correct use of `analysis_result` and `plot_data_df` (especially if plot generated)?
2. Data Analysis: Relevance to query/data? Accurate transformations/calculations? Appropriate methods? `plot_data_df` content match plot?
3. Plot Quality (if `{plot_image_path}` exists): Appropriate type? Well-labeled? Clear?
4. Report Quality: Clear, concise, insightful? Reflects `data_csv_content`? Addresses query? Accessible language?
5. Overall Effectiveness: How well query addressed (score 1-10)? Actionable suggestions for worker AI (esp. `plot_data_df`).
Critique:"""
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    st.session_state.selected_worker_model = st.selectbox("Select Worker Model:", AVAILABLE_MODELS,
                                                          index=AVAILABLE_MODELS.index(
                                                              st.session_state.selected_worker_model))
    st.session_state.selected_judge_model = st.selectbox("Select Judge Model:", AVAILABLE_MODELS,
                                                         index=AVAILABLE_MODELS.index(
                                                             st.session_state.selected_judge_model))

    st.header("📤 Upload CSV")
    uploaded_file = st.file_uploader("Select your CSV file:", type="csv")

    if uploaded_file is not None:
        if st.button("Process CSV & Get Suggestions", key="process_csv_btn"):
            with st.spinner("Processing CSV and starting CDO Workflow..."):
                if load_csv_and_get_summary(uploaded_file):
                    st.success(f"CSV '{st.session_state.data_source_name}' processed.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"Processed '{st.session_state.data_source_name}'. Starting CDO initial data description with **{st.session_state.selected_worker_model}**..."})
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"Uploaded {st.session_state.data_source_name} for CDO workflow."},
                        {"output": "CSV processed. Requesting CDO initial description."})
                    st.session_state.current_analysis_artifacts = {}
                    st.session_state.cdo_initial_report_text = None

                    worker_llm = get_llm_instance(st.session_state.selected_worker_model)
                    if not worker_llm or not st.session_state.data_summary:
                        st.error("Worker LLM or data summary not available.")
                        st.rerun()

                    cdo_initial_report_text = ""
                    with st.spinner(
                            f"CDO ({st.session_state.selected_worker_model}) is performing initial data description..."):
                        try:
                            memory_context = st.session_state.lc_memory.load_memory_variables({})
                            cdo_desc_prompt_inputs = {
                                "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                "chat_history": memory_context.get("chat_history", "")
                            }
                            formatted_cdo_desc_prompt = cdo_initial_data_description_prompt_template.format_prompt(
                                **cdo_desc_prompt_inputs)
                            response_obj = worker_llm.invoke(formatted_cdo_desc_prompt)
                            cdo_initial_report_text = response_obj.content if hasattr(response_obj,
                                                                                      'content') else response_obj.get(
                                'text', "Error: CDO description failed.")
                            st.session_state.cdo_initial_report_text = cdo_initial_report_text
                            st.session_state.messages.append({"role": "assistant",
                                                              "content": f"**CDO's Initial Data Description & Quality Assessment (via {st.session_state.selected_worker_model}):**\n\n{cdo_initial_report_text}\n\nNow, other department heads will provide their perspectives based on this."})
                            st.session_state.lc_memory.save_context(
                                {"user_query": "System requested CDO initial data description."},
                                {"output": f"CDO provided initial report: {cdo_initial_report_text[:200]}..."})
                        except Exception as e:
                            st.error(f"Error during CDO initial data description: {e}")
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"Error in CDO initial description: {e}"})
                            cdo_initial_report_text = "Error generating CDO initial report."
                            st.session_state.cdo_initial_report_text = cdo_initial_report_text
                    other_perspectives_text = ""
                    if "Error generating CDO initial report." not in cdo_initial_report_text and cdo_initial_report_text:
                        with st.spinner(
                                f"Department Heads (via {st.session_state.selected_worker_model}) are discussing based on CDO's report..."):
                            try:
                                memory_context = st.session_state.lc_memory.load_memory_variables({})
                                perspectives_prompt_inputs = {
                                    "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                    "chat_history": memory_context.get("chat_history", ""),
                                    "cdo_initial_report": cdo_initial_report_text
                                }
                                formatted_perspectives_prompt = individual_perspectives_prompt_template.format_prompt(
                                    **perspectives_prompt_inputs)
                                response_obj = worker_llm.invoke(formatted_perspectives_prompt)
                                other_perspectives_text = response_obj.content if hasattr(response_obj,
                                                                                          'content') else response_obj.get(
                                    'text', "Error: Perspectives generation failed.")
                                st.session_state.messages.append({"role": "assistant",
                                                                  "content": f"**Departmental Perspectives (informed by CDO's report, via {st.session_state.selected_worker_model}):**\n\n{other_perspectives_text}\n\nNext, the CDO will synthesize the final strategy."})
                                st.session_state.lc_memory.save_context(
                                    {"user_query": "System requested VPs' perspectives after CDO report."},
                                    {"output": f"VPs provided perspectives: {other_perspectives_text[:200]}..."})
                            except Exception as e:
                                st.error(f"Error getting departmental perspectives: {e}")
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": f"Error getting perspectives: {e}"})
                                other_perspectives_text = "Error generating VPs' perspectives."

                    if "Error" not in cdo_initial_report_text and "Error" not in other_perspectives_text and other_perspectives_text:
                        with st.spinner(
                                f"CDO ({st.session_state.selected_worker_model}) is synthesizing the final analysis strategy..."):
                            try:
                                memory_context = st.session_state.lc_memory.load_memory_variables({})
                                synthesis_prompt_inputs = {
                                    "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                    "chat_history": memory_context.get("chat_history", ""),
                                    "cdo_initial_report": cdo_initial_report_text,
                                    "generated_perspectives_from_others": other_perspectives_text
                                }
                                formatted_synthesis_prompt = synthesize_analysis_suggestions_prompt_template.format_prompt(
                                    **synthesis_prompt_inputs)
                                response_obj = worker_llm.invoke(formatted_synthesis_prompt)
                                strategy_text = response_obj.content if hasattr(response_obj,
                                                                                'content') else response_obj.get('text',
                                                                                                                 "Error: Strategy synthesis failed.")
                                st.session_state.messages.append({"role": "assistant",
                                                                  "content": f"**Final 5 Analysis Strategy Suggestions (by CDO - {st.session_state.selected_worker_model}):**\n\n{strategy_text}\n\nWhat would you like to analyze further?"})
                                st.session_state.lc_memory.save_context(
                                    {"user_query": "System requested CDO final strategy synthesis."},
                                    {"output": f"CDO provided final strategy: {strategy_text[:200]}..."})
                            except Exception as e:
                                st.error(f"Error synthesizing final strategy: {e}")
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": f"Error in CDO final strategy synthesis: {e}"})
                    elif not other_perspectives_text and "Error" not in cdo_initial_report_text:
                        st.warning("Skipping final strategy synthesis as departmental perspectives were not generated.")
                    else:
                        st.warning("Skipping further steps due to errors in earlier stages of CDO workflow.")
                    st.rerun()
                else:
                    st.error("Failed to process CSV.")

    if st.session_state.current_dataframe is not None:
        st.subheader("File Loaded:")
        st.write(
            f"**{st.session_state.data_source_name}** ({len(st.session_state.current_dataframe)} rows x {len(st.session_state.current_dataframe.columns)} columns)")
        with st.expander("View Data Summary & Details"):
            if st.session_state.data_summary:
                st.json(st.session_state.data_summary)
            else:
                st.write("No data summary available.")
        with st.expander("View DataFrame Head (First 5 rows)"):
            st.dataframe(st.session_state.current_dataframe.head())
        if st.button("Clear Loaded Data & Chat", key="clear_data_btn"):
            keys_to_reset = ["current_dataframe", "data_summary", "data_source_name", "current_analysis_artifacts",
                             "messages", "lc_memory", "cdo_initial_report_text", "trigger_report_generation",
                             "report_target_data_path", "report_target_plot_path", "report_target_query",
                             "trigger_judging"]
            for key in keys_to_reset:
                if key in st.session_state: del st.session_state[key]
            st.session_state.messages = [
                {"role": "assistant", "content": "Data and chat reset. Upload a new CSV file."}]
            st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                                  input_key="user_query")
            st.session_state.current_analysis_artifacts = {}
            st.session_state.cdo_initial_report_text = None
            cleaned_files_count = 0
            for item in os.listdir(TEMP_DATA_STORAGE):
                if item.endswith((".png", ".csv", ".txt", ".jpg", ".jpeg", ".svg")):
                    try:
                        os.remove(os.path.join(TEMP_DATA_STORAGE, item))
                        cleaned_files_count += 1
                    except Exception as e:
                        st.warning(f"Could not remove temp file {item}: {e}")
            st.success(f"Data, chat, and {cleaned_files_count} temporary files cleared.")
            st.rerun()

    st.markdown("---")
    st.info(
        f"Worker Model: **{st.session_state.selected_worker_model}**\n\nJudge Model: **{st.session_state.selected_judge_model}**")
    st.info(f"Temporary files stored in: `{os.path.abspath(TEMP_DATA_STORAGE)}`")
    st.warning(
        "⚠️ **Security Note:** This application uses `exec()` to run AI-generated Python code. This is for demonstration purposes ONLY.")

# --- Main Chat Interface ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "executed_result" in message:
            executed_res = message["executed_result"]
            res_type = executed_res.get("type")
            original_query = message.get("original_user_query",
                                         st.session_state.current_analysis_artifacts.get("original_user_query",
                                                                                         "Unknown query"))
            if res_type == "table":
                try:
                    df_to_display = pd.read_csv(executed_res["data_path"])
                    st.dataframe(df_to_display)
                    if st.button(f"📊 Generate Report for this Table##{i}", key=f"report_table_btn_{i}"):
                        st.session_state.trigger_report_generation = True
                        st.session_state.report_target_data_path = executed_res["data_path"]
                        st.session_state.report_target_plot_path = None
                        st.session_state.report_target_query = original_query
                        st.rerun()
                except Exception as e:
                    st.error(f"Error displaying table from {executed_res['data_path']}: {e}")
            elif res_type == "plot":
                if os.path.exists(executed_res["plot_path"]):
                    st.image(executed_res["plot_path"])
                    if executed_res.get("data_path") and os.path.exists(executed_res["data_path"]):
                        if st.button(f"📄 Generate Report for Plot Data##{i}", key=f"report_plot_data_btn_{i}"):
                            st.session_state.trigger_report_generation = True
                            st.session_state.report_target_data_path = executed_res["data_path"]
                            st.session_state.report_target_plot_path = executed_res["plot_path"]
                            st.session_state.report_target_query = original_query
                            st.rerun()
                    else:
                        st.caption("Note: Specific data table for this plot was not saved or found.")
                        if st.button(f"📄 Generate Descriptive Report for Plot Image##{i}",
                                     key=f"report_plot_desc_btn_{i}"):
                            st.session_state.trigger_report_generation = True
                            st.session_state.report_target_data_path = None
                            st.session_state.report_target_plot_path = executed_res["plot_path"]
                            st.session_state.report_target_query = original_query
                            st.rerun()
                else:
                    st.warning(f"Plot image not found: {executed_res['plot_path']}")
            elif res_type == "text":
                st.markdown(f"**Execution Output:**\n```\n{executed_res.get('value', 'No textual output.')}\n```")
            elif res_type == "report_generated":
                if executed_res.get("report_path") and os.path.exists(executed_res["report_path"]):
                    st.markdown(f"_Report saved to: `{os.path.abspath(executed_res['report_path'])}`_")

            artifacts_for_judging = st.session_state.get("current_analysis_artifacts", {})
            can_judge = artifacts_for_judging.get("generated_code") and (
                        artifacts_for_judging.get("executed_data_path") or artifacts_for_judging.get(
                    "plot_image_path") or artifacts_for_judging.get("executed_text_output") or (
                                    res_type == "text" and executed_res.get("value")))
            if can_judge:  # Show judge button if code was generated and there's some output
                if st.button(f"⚖️ Judge this Analysis by {st.session_state.selected_judge_model}##{i}",
                             key=f"judge_btn_{i}"):
                    st.session_state.trigger_judging = True
                    st.rerun()

        if message["role"] == "assistant" and "critique_text" in message:
            with st.expander(f"View Critique by {st.session_state.selected_judge_model}", expanded=True):
                st.markdown(message["critique_text"])

            # Add PDF export button after critique is shown
            if st.button(f"📄 Export Full Analysis to PDF##{i}", key=f"pdf_export_btn_{i}"):
                with st.spinner("Generating PDF report..."):
                    pdf_file_path = export_analysis_to_pdf(st.session_state.current_analysis_artifacts)
                    if pdf_file_path and os.path.exists(pdf_file_path):
                        with open(pdf_file_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download Analysis PDF",
                                data=pdf_file,
                                file_name=os.path.basename(pdf_file_path),
                                mime="application/pdf",
                                key=f"download_pdf_{i}"
                            )
                        st.success(f"PDF report generated: {os.path.basename(pdf_file_path)}")
                    else:
                        st.error("Failed to generate PDF report.")

# --- Report Generation Logic ---
if st.session_state.get("trigger_report_generation", False):
    st.session_state.trigger_report_generation = False
    data_path_for_report = st.session_state.get("report_target_data_path")
    plot_path_for_report = st.session_state.get("report_target_plot_path")
    query_that_led_to_data = st.session_state.report_target_query
    worker_llm = get_llm_instance(st.session_state.selected_worker_model)

    if not worker_llm or not st.session_state.data_summary or (not data_path_for_report and not plot_path_for_report):
        st.error("Cannot generate report: LLM, data summary, or target data/plot path missing.")
        st.rerun()

    csv_content_for_report = "N/A - Report is likely descriptive of a plot image."
    if data_path_for_report and os.path.exists(data_path_for_report):
        try:
            with open(data_path_for_report, 'r', encoding='utf-8') as f:
                csv_content_for_report = f.read()
        except Exception as e:
            st.error(f"Error reading data file ('{data_path_for_report}') for report: {e}")
            st.rerun()
    elif plot_path_for_report:
        st.info("Generating a descriptive report for the plot image as specific data table was not provided.")

    with st.chat_message("assistant"):
        report_spinner_msg_container = st.empty()
        report_spinner_msg_container.markdown(
            f"📝 **{st.session_state.selected_worker_model}** is generating report for: '{query_that_led_to_data}'...")
        with st.spinner("Generating report..."):
            try:
                memory_context = st.session_state.lc_memory.load_memory_variables({})
                report_prompt_inputs = {
                    "table_data_csv": csv_content_for_report,
                    "original_data_summary": json.dumps(st.session_state.data_summary, indent=2),
                    "user_query_that_led_to_data": query_that_led_to_data,
                    "chat_history": memory_context.get("chat_history", "")
                }
                formatted_prompt = report_generation_prompt_template.format_prompt(**report_prompt_inputs)
                response_obj = worker_llm.invoke(formatted_prompt)
                report_text = response_obj.content if hasattr(response_obj, 'content') else response_obj.get('text',
                                                                                                             "Error: Report generation failed.")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query_part = "".join(c if c.isalnum() else "_" for c in query_that_led_to_data[:30])
                filepath = os.path.join(TEMP_DATA_STORAGE, f"report_for_{safe_query_part}_{timestamp}.txt")
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(report_text)

                st.session_state.current_analysis_artifacts["generated_report_path"] = filepath
                st.session_state.current_analysis_artifacts["report_query"] = query_that_led_to_data
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"📊 **Report by {st.session_state.selected_worker_model} for '{query_that_led_to_data}':**\n\n{report_text}",
                    "original_user_query": query_that_led_to_data,
                    "executed_result": {"type": "report_generated", "report_path": filepath,
                                        "data_source_path": data_path_for_report or "N/A",
                                        "plot_source_path": plot_path_for_report or "N/A"}
                })
                st.session_state.lc_memory.save_context(
                    {"user_query": f"Requested report for: '{query_that_led_to_data}'"},
                    {"output": f"Report generated: {report_text[:100]}..."})
                report_spinner_msg_container.empty()
                st.rerun()
            except Exception as e:
                st.error(f"Error generating report: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error generating report: {e}"})
                if 'report_spinner_msg_container' in locals() and report_spinner_msg_container: report_spinner_msg_container.empty()
                st.rerun()
    for key in ["report_target_data_path", "report_target_plot_path", "report_target_query"]:
        if key in st.session_state: del st.session_state[key]

# --- Judging Logic ---
if st.session_state.get("trigger_judging", False):
    st.session_state.trigger_judging = False
    artifacts = st.session_state.current_analysis_artifacts
    judge_llm = get_llm_instance(st.session_state.selected_judge_model)
    original_query_for_artifacts = artifacts.get("original_user_query", "Unknown query for artifacts")

    if not judge_llm or not artifacts.get("generated_code"):
        st.error("Judge LLM not available or generated code missing for critique.")
        st.rerun()

    try:
        code_content = artifacts.get("generated_code", "No code found.")
        data_content = "No data file produced or found."
        if artifacts.get("executed_data_path") and os.path.exists(artifacts["executed_data_path"]):
            with open(artifacts["executed_data_path"], 'r', encoding='utf-8') as f:
                data_content = f.read()
        elif artifacts.get("executed_text_output"):
            data_content = f"Text output: {artifacts.get('executed_text_output')}"

        report_content = "No report generated or found."
        if artifacts.get("generated_report_path") and os.path.exists(artifacts["generated_report_path"]):
            with open(artifacts["generated_report_path"], 'r', encoding='utf-8') as f:
                report_content = f.read()
        elif artifacts.get("report_query") and not artifacts.get("generated_report_path"):
            report_content = f"Report expected for '{artifacts.get('report_query')}' but not found."

        plot_image_actual_path = artifacts.get("plot_image_path", "N/A")
        plot_info_for_judge = f"Plot Image: {plot_image_actual_path}."
        if plot_image_actual_path == "N/A":
            plot_info_for_judge = "No plot generated."
        elif not os.path.exists(plot_image_actual_path):
            plot_info_for_judge = f"Plot at '{plot_image_actual_path}' not found."
        else:
            plot_info_for_judge = f"Plot at '{plot_image_actual_path}'. "
            if artifacts.get("executed_data_path") and "plot_data_for" in os.path.basename(
                    artifacts.get("executed_data_path", "")):
                plot_info_for_judge += f"Plot-specific data at '{artifacts.get('executed_data_path')}'."
            else:
                plot_info_for_judge += "Plot-specific data not found/applicable."

        with st.chat_message("assistant"):
            critique_spinner_msg_container = st.empty()
            critique_spinner_msg_container.markdown(
                f"⚖️ **{st.session_state.selected_judge_model}** is critiquing analysis for: '{original_query_for_artifacts}'...")
            with st.spinner("Generating critique..."):
                judging_inputs = {
                    "python_code": code_content, "data_csv_content": data_content,
                    "report_text_content": report_content, "original_user_query": original_query_for_artifacts,
                    "data_summary": json.dumps(st.session_state.data_summary,
                                               indent=2) if st.session_state.data_summary else "N/A",
                    "plot_image_path": plot_image_actual_path, "plot_info": plot_info_for_judge
                }
                formatted_judging_prompt = judging_prompt_template.format_prompt(**judging_inputs)
                critique_response_obj = judge_llm.invoke(formatted_judging_prompt)
                critique_text = critique_response_obj.content if hasattr(critique_response_obj,
                                                                         'content') else critique_response_obj.get(
                    'text', "Error: Critique generation failed.")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query_part = "".join(c if c.isalnum() else "_" for c in original_query_for_artifacts[:30])
                critique_filename = f"critique_on_{safe_query_part}_{timestamp}.txt"
                critique_filepath = os.path.join(TEMP_DATA_STORAGE, critique_filename)
                with open(critique_filepath, "w", encoding='utf-8') as f: f.write(critique_text)

                # Store critique path in artifacts
                st.session_state.current_analysis_artifacts["generated_critique_path"] = critique_filepath

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚖️ **Critique by {st.session_state.selected_judge_model} for '{original_query_for_artifacts}' (saved to `{os.path.abspath(critique_filepath)}`):**",
                    "critique_text": critique_text  # This will be displayed by the chat loop, then PDF button
                })
                st.session_state.lc_memory.save_context(
                    {"user_query": f"Requested critique for: '{original_query_for_artifacts}'"},
                    {"output": f"Critique generated: {critique_text[:100]}..."})
                critique_spinner_msg_container.empty()
                st.rerun()
    except Exception as e:
        st.error(f"Error during critique generation: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error generating critique: {e}"})
        if 'critique_spinner_msg_container' in locals() and critique_spinner_msg_container: critique_spinner_msg_container.empty()
        st.rerun()

# --- Chat Input and Code Generation/Execution Logic ---
if user_query := st.chat_input("Ask for analysis (Worker Model will generate and run code)...", key="user_query_input"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if st.session_state.current_dataframe is None or st.session_state.data_summary is None:
        st.warning("Please upload and process a CSV file first.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "I need CSV data. Please upload a file first."})
        st.rerun()

    worker_llm = get_llm_instance(st.session_state.selected_worker_model)
    if not worker_llm:
        st.error(f"Worker model {st.session_state.selected_worker_model} not initialized. Check API key/selection.")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Worker LLM ({st.session_state.selected_worker_model}) unavailable."})
        st.rerun()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        generated_code_string = ""
        st.session_state.current_analysis_artifacts = {"original_user_query": user_query}

        message_placeholder.markdown(
            f"⏳ **{st.session_state.selected_worker_model}** generating code for: '{user_query}'...")
        with st.spinner(f"{st.session_state.selected_worker_model} generating Python code..."):
            try:
                memory_context = st.session_state.lc_memory.load_memory_variables({})
                prompt_inputs = {
                    "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                    "user_query": user_query,
                    "chat_history": memory_context.get("chat_history", "")
                }
                formatted_prompt = code_generation_prompt_template.format_prompt(**prompt_inputs)
                response_obj = worker_llm.invoke(formatted_prompt)
                generated_code_string = response_obj.content if hasattr(response_obj, 'content') else response_obj.get(
                    'text', "")

                for prefix in ["```python\n", "```\n", "```"]:
                    if generated_code_string.startswith(prefix): generated_code_string = generated_code_string[
                                                                                         len(prefix):]
                if generated_code_string.endswith("\n```"):
                    generated_code_string = generated_code_string[:-len("\n```")]
                elif generated_code_string.endswith("```"):
                    generated_code_string = generated_code_string[:-len("```")]
                generated_code_string = generated_code_string.strip()

                st.session_state.current_analysis_artifacts["generated_code"] = generated_code_string
                assistant_base_content = f"🔍 **Generated Code by {st.session_state.selected_worker_model} for '{user_query}':**\n```python\n{generated_code_string}\n```\n"
                message_placeholder.markdown(assistant_base_content + "\n⏳ Executing code locally...")
            except Exception as e:
                error_msg = f"Error generating code: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.lc_memory.save_context({"user_query": user_query}, {"output": f"Code Gen Error: {e}"})
                st.rerun()

        if generated_code_string:
            current_assistant_response_message = {"role": "assistant", "content": assistant_base_content,
                                                  "original_user_query": user_query}
            with st.spinner("Executing code..."):
                execution_result = code_executor.execute_code(generated_code_string, st.session_state.current_dataframe)

                if execution_result.get("data_path"): st.session_state.current_analysis_artifacts[
                    "executed_data_path"] = execution_result["data_path"]
                if execution_result.get("plot_path"): st.session_state.current_analysis_artifacts["plot_image_path"] = \
                execution_result["plot_path"]
                if execution_result.get("type") == "text" and execution_result.get("value"):
                    st.session_state.current_analysis_artifacts["executed_text_output"] = execution_result.get("value")

                llm_memory_output = ""
                if execution_result["type"] == "error":
                    current_assistant_response_message[
                        "content"] += f"\n⚠️ **Execution Error:**\n```\n{execution_result['message']}\n```"
                    # If exec fails, capture the error in analysis_result if it's still the default
                    if st.session_state.current_analysis_artifacts.get("executed_text_output", "").startswith(
                            "Code executed, but 'analysis_result'"):
                        st.session_state.current_analysis_artifacts[
                            "executed_text_output"] = f"Execution Error: {execution_result.get('final_analysis_result_value', 'Unknown error during execution')}"
                    llm_memory_output = f"Exec Error: {execution_result['message'][:100]}..."


                else:  # Successful execution
                    current_assistant_response_message["content"] += "\n✅ **Code Executed Successfully!**"
                    current_assistant_response_message["executed_result"] = execution_result
                    # If the result is text and it's the default message, update the artifact to show this.
                    if execution_result.get("type") == "text" and \
                            str(execution_result.get("value", "")).startswith("Code executed, but 'analysis_result'"):
                        st.session_state.current_analysis_artifacts["executed_text_output"] = str(
                            execution_result.get("value", ""))
                    elif execution_result.get("type") == "text":
                        st.session_state.current_analysis_artifacts["executed_text_output"] = str(
                            execution_result.get("value", ""))

                    if execution_result.get("data_path"): current_assistant_response_message[
                        "content"] += f"\n💾 Data saved: `{os.path.abspath(execution_result['data_path'])}`"
                    if execution_result.get("plot_path"): current_assistant_response_message[
                        "content"] += f"\n🖼️ Plot saved: `{os.path.abspath(execution_result['plot_path'])}`"
                    if execution_result.get("data_path") and "plot_data_for" in os.path.basename(
                        execution_result.get("data_path", "")): current_assistant_response_message[
                        "content"] += " (Plot-specific data also saved)."

                    if execution_result["type"] == "table":
                        llm_memory_output = f"Table: {os.path.basename(execution_result['data_path'])}"
                    elif execution_result["type"] == "plot":
                        llm_memory_output = f"Plot: {os.path.basename(execution_result['plot_path'])}"
                        if execution_result.get(
                            "data_path"): llm_memory_output += f" (Data: {os.path.basename(execution_result['data_path'])})"
                    elif execution_result["type"] == "text":
                        llm_memory_output = f"Text: {execution_result['value'][:50]}..."
                    else:
                        llm_memory_output = "Code exec, unknown result type."

                st.session_state.lc_memory.save_context(
                    {"user_query": f"{user_query}\n---Code---\n{generated_code_string}\n---End Code---"},
                    {"output": llm_memory_output})
                message_placeholder.markdown(current_assistant_response_message["content"])
                st.session_state.messages.append(current_assistant_response_message)
                st.rerun()
