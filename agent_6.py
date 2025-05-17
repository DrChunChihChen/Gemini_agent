# simplified_data_agent_streamlit_v3_enhanced.py

import streamlit as st
import pandas as pd
import os
import io  # To read CSV content as string
import json  # For data summary
import datetime  # For timestamping saved files
import matplotlib.pyplot  # Explicit import for placeholder and execution scope
import seaborn  # Explicit import for placeholder and execution scope

# --- Langchain/LLM Components ---
# pip install langchain-google-genai langchain-core
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# --- Configuration ---
# --- Configuration ---
LLM_API_KEY = os.environ.get("LLM_API_KEY", "API_Key請輸入你的Gemini api")  # Use st.secrets for deployment
TEMP_DATA_STORAGE = "temp_data_simplified_agent/"
os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)

AVAILABLE_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
DEFAULT_WORKER_MODEL = "gemini-2.0-flash-lite"
DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"

if LLM_API_KEY == "YOUR_API_KEY_HERE":
    st.error(
        "Please set your LLM_API_KEY (e.g., GOOGLE_API_KEY) environment variable or Streamlit secret for full functionality.")


# --- LLM Initialization & Placeholder ---
class PlaceholderLLM:
    """Simulates LLM responses for when an API key is not available."""

    def __init__(self, model_name="placeholder_model"):
        self.model_name = model_name
        st.warning(f"Using PlaceholderLLM for {self.model_name} as API key is not set or invalid.")

    def invoke(self, prompt_input):
        prompt_str = str(prompt_input)

        if "panel of expert department heads" in prompt_str:  # New: Individual perspectives prompt
            return {"text": """
*Placeholder Individual Perspectives ({model_name}):*

**CEO (首席執行官 - Chief Executive Officer):**
* What are the overall sales trends compared to last quarter/year?
* Are we gaining or losing market share in key segments based on this data?
* What major risks or opportunities does this data highlight for our strategic goals?

**CFO (首席財務官 - Chief Financial Officer):**
* Which products/services are the most and least profitable according to this data?
* What are the major cost drivers visible here, and are they within budget?
* Can we identify any trends in revenue streams that require attention?

**CTO (首席技術官 - Chief Technology Officer):**
* How reliable and clean does this dataset appear to be? Any obvious gaps or inconsistencies?
* Are there opportunities to automate the collection or processing of this type of data?
* Does this data suggest any need for new tech tools or infrastructure for better analysis?

**COO (首席運營官 - Chief Operating Officer):**
* Where are the operational bottlenecks or inefficiencies suggested by this data (e.g., by region, product line)?
* How can we optimize resource allocation based on these findings?
* Are there any process improvements suggested by patterns in this data?

**CMO (首席行銷官 - Chief Marketing Officer):**
* What are the key customer segments identifiable from this data and their purchasing behavior?
* How effective do our recent marketing campaigns appear to be, if reflected here?
* What market trends or shifts in customer preference can be inferred?
""".format(model_name=self.model_name)}
        elif "The following individual perspectives and questions have been generated" in prompt_str:  # New: Synthesis prompt
            return {"text": """
*Placeholder Synthesized Suggestions (based on prior perspectives from {model_name}):*
1.  **Strategic Financial Review:** Analyze detailed sales trends (CEO) against product profitability metrics (CFO) to identify high-impact areas for growth.
2.  **Operational Tech Upgrade:** Investigate operational bottlenecks (COO) and assess data automation opportunities (CTO) to improve efficiency in those areas.
3.  **Market-Driven Growth:** Link customer segmentation insights (CMO) with overall market share goals (CEO) to refine marketing strategies.
4.  **Cost-Benefit Analysis for Innovation:** Explore cost drivers (CFO) related to potential tech-driven innovations (CTO) for strategic initiatives (CEO).
5.  **Data Quality for Campaign Efficacy:** Evaluate if current data quality and systems (CTO) are sufficient for accurately measuring marketing campaign effectiveness (CMO) and operational performance (COO).
""".format(model_name=self.model_name)}
        elif "Python code:" in prompt_str and "User Query:" in prompt_str:  # Code generation
            user_query_segment = prompt_str.split("User Query:")[1].split("\n")[0].lower()
            if "average sales" in user_query_segment:
                return {"text": "analysis_result = df['sales'].mean()\nplot_data_df = df[['sales']].copy()"}
            elif "plot" in user_query_segment:
                return {
                    "text": "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.hist(df.iloc[:, 0])\nplot_path = 'placeholder_plot.png'\nplt.savefig(plot_path)\nplt.close(fig)\nanalysis_result = plot_path\nplot_data_df = df.copy()"}
            else:
                return {"text": "analysis_result = df.head()\nplot_data_df = df.head()"}
        elif "Generate a textual report" in prompt_str:
            return {
                "text": f"### Placeholder Report ({self.model_name})\nThis is a placeholder report based on the provided data."}
        elif "Critique the following analysis artifacts" in prompt_str:  # Judging prompt
            return {"text": f"""
### Placeholder Critique ({self.model_name})
**Overall Assessment:** The worker AI's output seems to be a placeholder.
**Python Code:** Appears to be Python.
**Data:** Data was mentioned.
**Report:** A report was mentioned.
**Suggestions for Worker AI:** Ensure all outputs are comprehensive.
"""}
        else:
            return {
                "text": f"Placeholder response from {self.model_name} for unrecognized prompt: {prompt_str[:200]}..."}


def get_llm_instance(model_name: str):
    """Gets an LLM instance, caching it in session state. Uses Placeholder if API key is missing."""
    if not model_name:
        st.error("No model name provided for LLM initialization.")
        return None

    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    if model_name not in st.session_state.llm_cache:
        if not LLM_API_KEY or LLM_API_KEY == "YOUR_API_KEY_HERE":
            st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
        else:
            try:
                # Adjust temperature based on role (e.g. judge more creative, worker more precise)
                # For Gemini, temperature is a float between 0.0 and 1.0 (sometimes 2.0 depending on model version)
                # A temperature of 0.0 is deterministic. Higher values increase randomness.
                # For judge, slightly higher might be good for diverse critiques. For worker, lower for factual tasks.
                temperature = 0.6 if st.session_state.get("selected_judge_model", "") == model_name else 0.3
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=LLM_API_KEY,
                    temperature=temperature,
                    convert_system_message_to_human=True  # Important for some Langchain versions with Gemini
                )
                st.session_state.llm_cache[model_name] = llm
            except Exception as e:
                st.error(f"Failed to initialize Gemini LLM ({model_name}): {e}")
                st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
    return st.session_state.llm_cache[model_name]


# --- Data Handling ---
def load_csv_and_get_summary(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.current_dataframe = df
        st.session_state.data_source_name = uploaded_file.name
        st.session_state.current_analysis_artifacts = {}  # Reset artifacts on new data

        # Create a more detailed summary
        data_summary = {
            "source_name": uploaded_file.name,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values_per_column": df.isnull().sum().to_dict(),
            "descriptive_stats_sample": df.describe(include='all').to_dict() if not df.empty else "N/A",
            # include all for non-numeric too
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


# --- Local Code Execution Engine ---
class LocalCodeExecutionEngine:
    def execute_code(self, code_string, df_input):
        if df_input is None:
            return {"type": "error", "message": "No data loaded to execute code on."}

        local_scope = {'df': df_input.copy(), 'pd': pd, 'plt': matplotlib.pyplot, 'sns': seaborn}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_df_saved_path = None  # Path for CSV data used in plot

        try:
            # Ensure 'analysis_result' and 'plot_data_df' are initialized in the scope
            # This helps avoid NameError if the LLM doesn't define them under certain conditions
            local_scope[
                'analysis_result'] = "Code executed, but 'analysis_result' was not explicitly set by the script."
            local_scope['plot_data_df'] = None  # Initialize plot_data_df

            exec(code_string, globals(), local_scope)  # globals() provides access to imported modules like pd, plt

            analysis_result = local_scope.get('analysis_result')
            plot_data_df = local_scope.get('plot_data_df')  # Retrieve plot_data_df from scope

            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                return {"type": "error", "message": analysis_result}

            # Handling plots
            if isinstance(analysis_result, str) and analysis_result.endswith((".png", ".jpg", ".jpeg", ".svg")):
                relative_plot_path = os.path.basename(analysis_result)
                # Check if the plot path is absolute or relative and if it exists
                # The code might save it directly in TEMP_DATA_STORAGE or a local path
                potential_paths = [
                    os.path.join(TEMP_DATA_STORAGE, relative_plot_path),  # LLM was told to save here
                    analysis_result,  # LLM provided an absolute path
                    relative_plot_path  # LLM provided a relative path (e.g. if running locally)
                ]
                final_plot_path = None
                for p_path in potential_paths:
                    if os.path.exists(p_path):
                        final_plot_path = p_path
                        break

                if not final_plot_path:
                    return {"type": "error",
                            "message": f"Plot file '{relative_plot_path}' not found. Checked: {potential_paths}"}

                # Save the dataframe that was used for plotting, if provided
                if isinstance(plot_data_df, pd.DataFrame) and not plot_data_df.empty:
                    plot_data_filename = f"plot_data_for_{os.path.splitext(relative_plot_path)[0]}_{timestamp}.csv"
                    plot_data_df_saved_path = os.path.join(TEMP_DATA_STORAGE, plot_data_filename)
                    plot_data_df.to_csv(plot_data_df_saved_path, index=False)
                elif plot_data_df is not None:  # It was set but not a dataframe or empty
                    st.warning(
                        "`plot_data_df` was set by the script but is not a valid DataFrame. Not saving associated data.")

                return {"type": "plot", "plot_path": final_plot_path, "data_path": plot_data_df_saved_path}

            # Handling tables (DataFrames or Series)
            elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                if isinstance(analysis_result, pd.Series):
                    analysis_result = analysis_result.to_frame()  # Convert Series to DataFrame for consistency

                if analysis_result.empty:
                    return {"type": "text", "value": "The analysis resulted in an empty table."}

                saved_csv_filename = f"table_result_{timestamp}.csv"
                saved_csv_path = os.path.join(TEMP_DATA_STORAGE, saved_csv_filename)
                analysis_result.to_csv(saved_csv_path, index=False)
                return {"type": "table", "data_path": saved_csv_path}

            # Handling text results (including the default message if analysis_result wasn't set)
            else:
                return {"type": "text", "value": str(analysis_result)}
        except Exception as e:
            # Capture more details from the execution error
            import traceback
            tb_str = traceback.format_exc()
            return {"type": "error", "message": f"Error during code execution: {str(e)}\nTraceback:\n{tb_str}"}


code_executor = LocalCodeExecutionEngine()

# --- Streamlit App UI and Logic ---
st.set_page_config(page_title="AI CSV Analyst v3.1 (Perspectives)", layout="wide")
st.title("🤖 AI CSV Analyst v3.1 (Perspectives, Worker & Judge)")
st.caption(
    "Upload CSV, get multi-perspective insights, then actionable suggestions (Worker Model), generate code & reports, and finally get a critique (Judge Model)."
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Select your models and upload a CSV to get started."}]
if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None
if "data_source_name" not in st.session_state:
    st.session_state.data_source_name = None
if "current_analysis_artifacts" not in st.session_state:  # Stores code, paths to data/plot/report for judging
    st.session_state.current_analysis_artifacts = {}
if "selected_worker_model" not in st.session_state:
    st.session_state.selected_worker_model = DEFAULT_WORKER_MODEL
if "selected_judge_model" not in st.session_state:
    st.session_state.selected_judge_model = DEFAULT_JUDGE_MODEL

if "lc_memory" not in st.session_state:  # Langchain conversation memory
    st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                          input_key="user_query")  # input_key matches what we save

# --- Prompt Templates ---
# NEW: Prompt for individual perspectives
individual_perspectives_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history"],
    template="""You are a panel of expert department heads. A user has uploaded a CSV file.
Data Summary:
{data_summary}

Based on this data, provide a detailed perspective from each of the following roles.
For each role, outline 2-3 specific questions they would ask, analyses they would want to perform, or initial observations they would make.
Structure your response clearly, with each role's perspective under a bolded heading (e.g., **CEO Perspective:**).

* **CEO (首席執行官 - Chief Executive Officer):** (Focus: Overall business strategy, growth, market position, major risks)
* **CFO (首席財務官 - Chief Financial Officer):** (Focus: Financial health, profitability, cost analysis, revenue streams)
* **CTO (首席技術官 - Chief Technology Officer):** (Focus: Data integrity, system performance, tech-driven innovation, efficiency)
* **COO (首席運營官 - Chief Operating Officer):** (Focus: Operational efficiency, process optimization, resource allocation)
* **CMO (首席行銷官 - Chief Marketing Officer):** (Focus: Customer insights, market trends, campaign effectiveness, sales performance)

Conversation History (for context, if any):
{chat_history}

Detailed Perspectives from Department Heads:
"""
)

# MODIFIED: Prompt to synthesize suggestions based on prior perspectives
synthesize_analysis_suggestions_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "generated_perspectives"],
    template="""You are a strategic analysis consultant AI. A user has uploaded a CSV file.
Data Summary:
{data_summary}

The following individual perspectives and questions have been generated by different department heads:
--- BEGIN GENERATED PERSPECTIVES ---
{generated_perspectives}
--- END GENERATED PERSPECTIVES ---

Based *specifically* on these provided perspectives and the data summary, synthesize them into a concise list of **5 distinct and actionable analysis strategy suggestions** for the user.
Ensure your suggestions directly address or build upon the points raised in the {generated_perspectives}.
Present these 5 suggestions as a numbered list. Each suggestion should be a question or a task.

Conversation History (for context, if any):
{chat_history}

Final 5 Analysis Strategy Suggestions (Synthesized from above perspectives):
"""
)

code_generation_prompt_template = PromptTemplate(
    input_variables=["data_summary", "user_query", "chat_history"],
    template="""You are an expert Python data analysis assistant.
Data Summary:
{data_summary}

User Query: "{user_query}"

Previous Conversation (for context):
{chat_history}

Your task is to generate a Python script to perform the requested analysis on a pandas DataFrame named `df`.
1.  The result of the analysis should be stored in a variable named `analysis_result`.
2.  If the analysis involves creating a plot:
    a.  Save the plot to a file named 'analysis_plot.png' in the current working directory.
    b.  Set `analysis_result` to the string 'analysis_plot.png'.
    c.  Create a pandas DataFrame named `plot_data_df` containing the exact data used for the plot. If the plot uses the entire `df`, then `plot_data_df = df.copy()`.
3.  If the analysis results in a table (e.g., a filtered DataFrame, a summary table), `analysis_result` should be this DataFrame or Series.
4.  If the analysis results in a single textual or numerical value, `analysis_result` should be this value.
5.  Ensure all necessary libraries (like matplotlib.pyplot as plt, seaborn as sns) are imported within the script if used.
Do not include any explanations or markdown formatting around the code. Output only the raw Python code.

Python code:"""
)

report_generation_prompt_template = PromptTemplate(
    input_variables=["table_data_csv", "original_data_summary", "user_query_that_led_to_data", "chat_history"],
    template="""You are an insightful data analyst. Your task is to generate a brief, easy-to-understand textual report based on the provided data and context.

Original Data Summary (context of the source dataset):
{original_data_summary}

User Query that led to this specific data table/plot: "{user_query_that_led_to_data}"

Chat History (for additional context):
{chat_history}

Analysis Result Data (this is the data your report should focus on, in CSV format):
```csv
{table_data_csv}
```

Based on the "Analysis Result Data", the "User Query", and the "Original Data Summary", write a concise report.
The report should:
1.  Briefly state the purpose of the analysis (derived from the user query).
2.  Highlight key findings from the "Analysis Result Data".
3.  Offer 1-2 insights or interpretations based on these findings.
4.  Be written in clear, accessible language. Avoid overly technical jargon.
Do not refer to the CSV data directly (e.g., "the CSV shows"), but rather interpret what the data means.

Report:"""
)

judging_prompt_template = PromptTemplate(
    input_variables=["python_code", "data_csv_content", "report_text_content", "original_user_query", "data_summary",
                     "plot_image_path", "plot_info"],  # Added plot_info
    template="""You are an expert data science reviewer and critique AI.
Your task is to meticulously evaluate the analysis artifacts produced by another AI assistant in response to a user query.

Original User Query: "{original_user_query}"
Original Data Summary (of the input dataset): {data_summary}

--- ARTIFACTS FOR REVIEW ---
1.  Generated Python Code:
    ```python
    {python_code}
    ```

2.  Data Produced by Code (if applicable, content in CSV format):
    ```csv
    {data_csv_content}
    ```
    {plot_info}  3.  Generated Report (if applicable):
    ```text
    {report_text_content}
    ```
--- END ARTIFACTS ---

Please provide a structured critique covering the following aspects:
1.  **Code Quality & Correctness:**
    * Does the code correctly implement the logic to address the user query?
    * Is it efficient and readable?
    * Does it follow Python best practices? Any bugs or potential issues?
    * Does it correctly use `analysis_result` and `plot_data_df` as instructed?

2.  **Data Analysis & Relevance:**
    * Is the analysis performed relevant to the user's query and the nature of the data?
    * Are the transformations and calculations accurate?
    * Is the choice of analysis methods appropriate?

3.  **Plot Quality (if a plot was generated and `plot_info` indicates its presence):**
    * Assuming the plot image at `{plot_image_path}` is viewable, is the plot type appropriate for the data and query?
    * Is it well-labeled (title, axes, legends)?
    * Does it effectively convey information? Is it clear or cluttered?

4.  **Report Quality (if a report was generated):**
    * Is the report clear, concise, and insightful?
    * Does it accurately reflect the data produced by the code?
    * Does it adequately address the user query and provide meaningful interpretations?
    * Is the language accessible?

5.  **Overall Effectiveness & Adherence to Query:**
    * How well did the worker AI, across all artifacts (code, data output, plot, report), address the user's original query?
    * Provide an overall score (1-10, where 10 is excellent).
    * Offer specific, actionable suggestions for the worker AI to improve in future, similar tasks.

Critique:"""
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    st.session_state.selected_worker_model = st.selectbox(
        "Select Worker Model:", AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_worker_model)
    )
    st.session_state.selected_judge_model = st.selectbox(
        "Select Judge Model:", AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_judge_model)
    )

    st.header("📤 Upload CSV")
    uploaded_file = st.file_uploader("Select your CSV file:", type="csv")

    if uploaded_file is not None:
        if st.button("Process CSV & Get Suggestions"):
            with st.spinner("Processing CSV..."):
                if load_csv_and_get_summary(uploaded_file):
                    st.success(f"CSV '{st.session_state.data_source_name}' processed.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Processed '{st.session_state.data_source_name}'. Now, let's get some initial perspectives from different department heads using {st.session_state.selected_worker_model}..."
                    })
                    st.session_state.lc_memory.save_context(
                        {
                            "user_query": f"Uploaded {st.session_state.data_source_name} and requested initial perspectives."},
                        {"output": "Processing CSV and preparing for perspective generation."}
                    )
                    st.session_state.current_analysis_artifacts = {}  # Clear previous artifacts

                    worker_llm = get_llm_instance(st.session_state.selected_worker_model)
                    if worker_llm and st.session_state.data_summary:
                        perspectives_text = ""
                        # STAGE 1: Get Individual Perspectives
                        with st.spinner(
                                f"🧠 {st.session_state.selected_worker_model} is consulting with department heads for initial perspectives..."):
                            try:
                                memory_context = st.session_state.lc_memory.load_memory_variables({})
                                perspectives_prompt_inputs = {
                                    "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                    "chat_history": memory_context.get("chat_history", "")
                                }
                                formatted_perspectives_prompt = individual_perspectives_prompt_template.format_prompt(
                                    **perspectives_prompt_inputs)
                                perspectives_response = worker_llm.invoke(formatted_perspectives_prompt.to_string())
                                perspectives_text = perspectives_response.content if hasattr(perspectives_response,
                                                                                             'content') else perspectives_response.get(
                                    'text', str(perspectives_response))

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"**Initial Perspectives from Department Heads (via {st.session_state.selected_worker_model}):**\n\n{perspectives_text}\n\nNow, synthesizing these into actionable suggestions..."
                                })
                                st.session_state.lc_memory.save_context(
                                    {"user_query": "System requested individual department perspectives."},
                                    {"output": f"Provided perspectives: {perspectives_text[:200]}..."}  # Log snippet
                                )
                            except Exception as e:
                                st.error(f"Error getting individual perspectives: {e}")
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": f"Error getting perspectives: {e}"})
                                perspectives_text = "Error generating perspectives."  # Ensure it has a value

                        # STAGE 2: Synthesize Suggestions based on Perspectives
                        if perspectives_text and "Error generating perspectives." not in perspectives_text:
                            with st.spinner(
                                    f"🛠️ {st.session_state.selected_worker_model} is synthesizing actionable suggestions..."):
                                try:
                                    memory_context = st.session_state.lc_memory.load_memory_variables(
                                        {})  # Reload for fresh history
                                    synthesis_prompt_inputs = {
                                        "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                        "chat_history": memory_context.get("chat_history", ""),
                                        "generated_perspectives": perspectives_text
                                    }
                                    formatted_synthesis_prompt = synthesize_analysis_suggestions_prompt_template.format_prompt(
                                        **synthesis_prompt_inputs)
                                    synthesis_response = worker_llm.invoke(formatted_synthesis_prompt.to_string())
                                    strategy_text = synthesis_response.content if hasattr(synthesis_response,
                                                                                          'content') else synthesis_response.get(
                                        'text', str(synthesis_response))

                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"**5 Actionable Analysis Strategy Suggestions for '{st.session_state.data_source_name}' (Synthesized by {st.session_state.selected_worker_model}):**\n\n{strategy_text}\n\nWhat would you like to analyze further based on these suggestions or your own query?"
                                    })
                                    st.session_state.lc_memory.save_context(
                                        {
                                            "user_query": "System requested synthesis of perspectives into 5 suggestions."},
                                        {"output": f"Provided 5 synthesized suggestions: {strategy_text[:200]}..."}
                                    )
                                except Exception as e:
                                    st.error(f"Error synthesizing suggestions: {e}")
                                    st.session_state.messages.append(
                                        {"role": "assistant", "content": f"Error synthesizing suggestions: {e}"})
                        else:
                            st.warning("Skipping synthesis of suggestions due to error in generating perspectives.")


                    else:
                        st.error("Worker LLM or data summary not available for generating suggestions.")
                    st.rerun()
                else:
                    st.error("Failed to process CSV.")

    if st.session_state.current_dataframe is not None:
        st.subheader("File Loaded:")
        st.write(f"{st.session_state.data_source_name} ({len(st.session_state.current_dataframe)} rows)")
        with st.expander("View Data Summary & Details"):
            st.json(st.session_state.data_summary)  # The summary is now more detailed
        with st.expander("View DataFrame Head (First 5 rows)"):
            st.dataframe(st.session_state.current_dataframe.head())

        if st.button("Clear Loaded Data & Chat"):
            # Reset relevant session state variables
            keys_to_reset = [
                "current_dataframe", "data_summary", "data_source_name",
                "current_analysis_artifacts", "messages", "lc_memory"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]

            # Re-initialize messages and memory
            st.session_state.messages = [
                {"role": "assistant", "content": "Data and chat reset. Upload a new CSV file."}]
            st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                                  input_key="user_query")

            # Clean up temporary files
            for item in os.listdir(TEMP_DATA_STORAGE):
                if item.endswith((".png", ".csv", ".txt", ".jpg", ".jpeg", ".svg")):  # Added more extensions
                    try:
                        os.remove(os.path.join(TEMP_DATA_STORAGE, item))
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {item}: {e}")
            st.success("Data, chat, and temporary files cleared.")
            st.rerun()

    st.markdown("---")
    st.info(
        f"Worker Model: **{st.session_state.selected_worker_model}**\n\nJudge Model: **{st.session_state.selected_judge_model}**")
    st.info(f"Temporary files are stored in: `{os.path.abspath(TEMP_DATA_STORAGE)}`")
    st.warning(
        "⚠️ **Security Note:** This application uses `exec()` to run AI-generated Python code. This is for demonstration purposes ONLY and can be a security risk if used with untrusted code or in a production environment.")

# --- Main Chat Interface ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Displaying artifacts and buttons for report/judging
        if message["role"] == "assistant" and "executed_result" in message:
            executed_res = message["executed_result"]
            res_type = executed_res.get("type")
            # original_query should be reliably present if code was executed
            original_query = message.get("original_user_query", "Unknown query that led to this result")

            if res_type == "table":
                try:
                    df_to_display = pd.read_csv(executed_res["data_path"])
                    st.dataframe(df_to_display)
                    # Button to generate report for this table
                    if st.button(f"📊 Generate Report for this Table##{i}", key=f"report_table_btn_{i}"):
                        st.session_state.trigger_report_generation = True
                        st.session_state.report_target_data_path = executed_res["data_path"]
                        st.session_state.report_target_query = original_query  # Pass the query
                        st.rerun()
                except Exception as e:
                    st.error(f"Error displaying table from {executed_res['data_path']}: {e}")

            elif res_type == "plot":
                if os.path.exists(executed_res["plot_path"]):
                    st.image(executed_res["plot_path"])
                    # If there's associated data with the plot, allow report generation
                    if executed_res.get("data_path") and os.path.exists(executed_res["data_path"]):
                        if st.button(f"📄 Generate Report for Plot Data##{i}", key=f"report_plot_data_btn_{i}"):
                            st.session_state.trigger_report_generation = True
                            st.session_state.report_target_data_path = executed_res["data_path"]
                            st.session_state.report_target_query = original_query  # Pass the query
                            st.rerun()
                    else:
                        # Offer to generate a report describing the plot itself if no data CSV
                        if st.button(f"📄 Generate Descriptive Report for Plot##{i}", key=f"report_plot_desc_btn_{i}"):
                            st.session_state.trigger_report_generation = True
                            # No data_path, report will be more descriptive of the plot image context
                            st.session_state.report_target_data_path = None
                            st.session_state.report_target_plot_path = executed_res["plot_path"]  # Pass plot path
                            st.session_state.report_target_query = original_query
                            st.rerun()

                else:
                    st.warning(f"Plot image not found at path: {executed_res['plot_path']}")

            elif res_type == "text":  # Simple text output from code execution
                st.markdown(f"**Execution Output:**\n```\n{executed_res.get('value', 'No textual output.')}\n```")

            elif res_type == "report_generated":  # Special type for after report is made
                if executed_res.get("report_path") and os.path.exists(executed_res["report_path"]):
                    st.markdown(f"_Report saved to: `{os.path.abspath(executed_res['report_path'])}`_")

            # Common "Judge this Analysis" button if code was generated
            # current_analysis_artifacts should be populated by the code execution step
            artifacts_for_judging = st.session_state.get("current_analysis_artifacts", {})
            can_judge = artifacts_for_judging.get("generated_code") and \
                        (artifacts_for_judging.get("executed_data_path") or artifacts_for_judging.get(
                            "plot_image_path") or executed_res.get("type") == "text")

            if can_judge:
                if st.button(f"⚖️ Judge this Analysis by {st.session_state.selected_judge_model}##{i}",
                             key=f"judge_btn_{i}"):
                    st.session_state.trigger_judging = True
                    # The original query that led to code generation is already in current_analysis_artifacts
                    st.rerun()

        # Displaying critique if it exists in the message
        if message["role"] == "assistant" and "critique_text" in message:
            with st.expander(f"View Critique by {st.session_state.selected_judge_model}", expanded=True):
                st.markdown(message["critique_text"])

# --- Report Generation Logic ---
if st.session_state.get("trigger_report_generation", False):
    st.session_state.trigger_report_generation = False  # Reset trigger
    data_path_for_report = st.session_state.get("report_target_data_path")
    plot_path_for_report = st.session_state.get("report_target_plot_path")  # For plot-descriptive reports
    query_that_led_to_data = st.session_state.report_target_query

    worker_llm = get_llm_instance(st.session_state.selected_worker_model)

    # Ensure necessary components are available
    if not worker_llm or not st.session_state.data_summary:
        st.error("Cannot generate report: LLM or original data summary missing.")
        st.rerun()
    if not data_path_for_report and not plot_path_for_report:  # Must have at least one
        st.error("Cannot generate report: Target data or plot path missing.")
        st.rerun()

    csv_content_for_report = "N/A - Report is likely descriptive of a plot image."
    if data_path_for_report and os.path.exists(data_path_for_report):
        try:
            with open(data_path_for_report, 'r', encoding='utf-8') as f:
                csv_content_for_report = f.read()
        except Exception as e:
            st.error(f"Error reading data file for report: {e}")
            st.rerun()  # Stop if data can't be read

    with st.chat_message("assistant"):
        report_spinner_msg_container = st.empty()
        report_spinner_msg_container.markdown(
            f"📝 {st.session_state.selected_worker_model} is generating a report for query: '{query_that_led_to_data}'...")

        with st.spinner("Generating report..."):
            try:
                memory_context = st.session_state.lc_memory.load_memory_variables({})
                report_prompt_inputs = {
                    "table_data_csv": csv_content_for_report,  # Will be "N/A..." if plot-only report
                    "original_data_summary": json.dumps(st.session_state.data_summary, indent=2),
                    "user_query_that_led_to_data": query_that_led_to_data,
                    "chat_history": memory_context.get("chat_history", "")
                }
                # If it's a plot-descriptive report, adjust prompt or add info
                # For now, the same prompt is used; it should be robust enough if table_data_csv is "N/A"
                # and the user query mentions a plot.

                formatted_prompt = report_generation_prompt_template.format_prompt(**report_prompt_inputs)
                response = worker_llm.invoke(formatted_prompt.to_string())
                report_text = response.content if hasattr(response, 'content') else response.get('text', str(response))

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"report_for_{query_that_led_to_data[:20].replace(' ', '_')}_{timestamp}.txt"
                filepath = os.path.join(TEMP_DATA_STORAGE, base_filename)
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(report_text)

                # Update current_analysis_artifacts with the generated report
                st.session_state.current_analysis_artifacts["generated_report_path"] = filepath
                st.session_state.current_analysis_artifacts[
                    "report_query"] = query_that_led_to_data  # The query that this report addresses

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"📊 **Report by {st.session_state.selected_worker_model} for query '{query_that_led_to_data}':**\n\n{report_text}",
                    "original_user_query": query_that_led_to_data,  # Keep track of original query for this message
                    "executed_result": {  # This helps the UI know a report was just made
                        "type": "report_generated",
                        "report_path": filepath,
                        "data_source_path": data_path_for_report if data_path_for_report else "N/A (Plot Description)",
                        "plot_source_path": plot_path_for_report if plot_path_for_report else "N/A"
                    }
                })
                st.session_state.lc_memory.save_context(
                    {"user_query": f"Requested report for analysis related to: '{query_that_led_to_data}'"},
                    {"output": f"Report generated: {report_text[:100]}..."}
                )
                report_spinner_msg_container.empty()
                st.rerun()
            except Exception as e:
                st.error(f"Error generating report: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error generating report: {e}"})
                report_spinner_msg_container.empty()  # Clear spinner message on error
                st.rerun()
    # Clear plot_path_for_report after use
    if "report_target_plot_path" in st.session_state:
        del st.session_state.report_target_plot_path

# --- Judging Logic ---
if st.session_state.get("trigger_judging", False):
    st.session_state.trigger_judging = False  # Reset trigger
    artifacts = st.session_state.current_analysis_artifacts
    judge_llm = get_llm_instance(st.session_state.selected_judge_model)

    # The original query that led to these artifacts should be in 'artifacts'
    original_query_for_artifacts = artifacts.get("original_user_query", "Unknown original query for these artifacts")

    if not judge_llm:
        st.error("Judge LLM not available. Cannot perform critique.")
        st.rerun()
    if not artifacts.get("generated_code"):
        st.error("Cannot perform critique: Generated Python code artifact is missing.")
        st.rerun()
    # It's okay if data or report is missing, critique should note that.

    try:
        code_content = artifacts.get("generated_code", "No code was generated or found.")

        data_content = "No specific data file (CSV) was produced by the code, or it was not found."
        if artifacts.get("executed_data_path") and os.path.exists(artifacts["executed_data_path"]):
            with open(artifacts["executed_data_path"], 'r', encoding='utf-8') as f:
                data_content = f.read()
        elif artifacts.get("executed_text_output"):  # If code produced text output directly
            data_content = f"Text output from code: {artifacts.get('executed_text_output')}"

        report_content = "No report was generated or found for this analysis cycle."
        if artifacts.get("generated_report_path") and os.path.exists(artifacts["generated_report_path"]):
            with open(artifacts["generated_report_path"], 'r', encoding='utf-f8') as f:  # Typo: utf-8
                report_content = f.read()
        elif artifacts.get("report_query") and not artifacts.get("generated_report_path"):
            report_content = f"A report was expected for the query '{artifacts.get('report_query')}' but not found."

        plot_image_actual_path = artifacts.get("plot_image_path", "N/A")
        plot_info_for_judge = f"Plot Image Path: {plot_image_actual_path}."
        if plot_image_actual_path == "N/A":
            plot_info_for_judge = "No plot image was generated or specified for this analysis."
        elif not os.path.exists(plot_image_actual_path):
            plot_info_for_judge = f"A plot was reportedly generated at '{plot_image_actual_path}', but the file is not found. Judge based on code if it intended to create one."

        with st.chat_message("assistant"):
            critique_spinner_msg_container = st.empty()
            critique_spinner_msg_container.markdown(
                f"⚖️ {st.session_state.selected_judge_model} is critiquing the analysis for query: '{original_query_for_artifacts}'...")
            with st.spinner("Generating critique..."):
                judging_inputs = {
                    "python_code": code_content,
                    "data_csv_content": data_content,  # This is content, not path
                    "report_text_content": report_content,  # This is content, not path
                    "original_user_query": original_query_for_artifacts,
                    "data_summary": json.dumps(st.session_state.data_summary,
                                               indent=2) if st.session_state.data_summary else "N/A",
                    "plot_image_path": plot_image_actual_path,  # Actual path for judge's context
                    "plot_info": plot_info_for_judge  # Descriptive text for the prompt
                }
                formatted_judging_prompt = judging_prompt_template.format_prompt(**judging_inputs)
                critique_response = judge_llm.invoke(formatted_judging_prompt.to_string())
                critique_text = critique_response.content if hasattr(critique_response,
                                                                     'content') else critique_response.get('text',
                                                                                                           str(critique_response))

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                critique_filename = f"critique_on_{original_query_for_artifacts[:20].replace(' ', '_')}_{timestamp}.txt"
                critique_filepath = os.path.join(TEMP_DATA_STORAGE, critique_filename)
                with open(critique_filepath, "w", encoding='utf-8') as f: f.write(critique_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚖️ **Critique by {st.session_state.selected_judge_model} for analysis of '{original_query_for_artifacts}' (saved to `{os.path.abspath(critique_filepath)}`):**",
                    "critique_text": critique_text  # This will be displayed by the chat loop
                })
                st.session_state.lc_memory.save_context(
                    {
                        "user_query": f"Requested critique for analysis related to query: '{original_query_for_artifacts}'"},
                    {"output": f"Critique generated: {critique_text[:100]}..."}
                )
                critique_spinner_msg_container.empty()
                st.rerun()
    except Exception as e:
        st.error(f"Error during critique generation: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error generating critique: {e}"})
        if 'critique_spinner_msg_container' in locals():
            critique_spinner_msg_container.empty()
        st.rerun()

# --- Chat Input and Code Generation/Execution Logic ---
if user_query := st.chat_input("Ask for analysis (Worker Model will generate and run code)..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Ensure data is loaded before attempting code generation
    if st.session_state.current_dataframe is None or st.session_state.data_summary is None:
        st.warning("Please upload and process a CSV file first before asking for analysis.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "I need CSV data to work with. Please upload a file first."})
        st.rerun()

    worker_llm = get_llm_instance(st.session_state.selected_worker_model)
    if not worker_llm:
        st.error(
            f"Worker model {st.session_state.selected_worker_model} could not be initialized. Please check your API key and model name selection.")
        st.session_state.messages.append({"role": "assistant",
                                          "content": f"Worker LLM ({st.session_state.selected_worker_model}) is not available. Cannot generate code."})
        st.rerun()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # For dynamic updates: Code -> Executing -> Result
        generated_code_string = ""
        assistant_base_content = ""  # Will hold the "Generated Code" part

        # Clear previous cycle's specific artifacts before generating new ones for this query
        # Keep general things like data_summary, dataframe.
        st.session_state.current_analysis_artifacts = {
            "original_user_query": user_query  # Store the current query that will lead to artifacts
        }

        message_placeholder.markdown(
            f"⏳ {st.session_state.selected_worker_model} is thinking and generating Python code for your query: '{user_query}'...")
        with st.spinner(f"{st.session_state.selected_worker_model} is generating Python code..."):
            try:
                memory_context = st.session_state.lc_memory.load_memory_variables({})
                prompt_inputs = {
                    "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                    "user_query": user_query,
                    "chat_history": memory_context.get("chat_history", "")
                }
                formatted_prompt = code_generation_prompt_template.format_prompt(**prompt_inputs)
                response = worker_llm.invoke(formatted_prompt.to_string())
                generated_code_string = response.content if hasattr(response, 'content') else response.get('text',
                                                                                                           str(response))

                # Clean up potential markdown code block fences
                if generated_code_string.startswith("```python"):
                    generated_code_string = generated_code_string[len("```python"):].strip()
                elif generated_code_string.startswith("```"):  # More general case
                    generated_code_string = generated_code_string[len("```"):].strip()
                if generated_code_string.endswith("```"):
                    generated_code_string = generated_code_string[:-len("```")].strip()

                st.session_state.current_analysis_artifacts["generated_code"] = generated_code_string
                # "original_user_query" already set when artifacts dict was initialized for this cycle

                assistant_base_content = f"🔍 **Generated Python Code by {st.session_state.selected_worker_model} for query '{user_query}':**\n```python\n{generated_code_string}\n```\n"
                message_placeholder.markdown(assistant_base_content + "\n⏳ Now executing this code locally...")

            except Exception as e:
                error_msg = f"Error generating code with {st.session_state.selected_worker_model}: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.lc_memory.save_context({"user_query": user_query},
                                                        {"output": f"Code Generation Error: {e}"})
                st.rerun()  # Stop if code generation fails

        if generated_code_string:  # Proceed to execution only if code was generated
            current_assistant_response_message = {
                "role": "assistant",
                "content": assistant_base_content,  # Starts with the generated code
                "original_user_query": user_query  # Important for linking judge/report buttons later
            }

            with st.spinner("Executing generated code locally... Please wait."):
                execution_result = code_executor.execute_code(generated_code_string, st.session_state.current_dataframe)

                # Update current_analysis_artifacts with paths from execution
                if execution_result.get("data_path"):  # CSV from table or plot data
                    st.session_state.current_analysis_artifacts["executed_data_path"] = execution_result["data_path"]
                if execution_result.get("plot_path"):  # Path to the plot image
                    st.session_state.current_analysis_artifacts["plot_image_path"] = execution_result["plot_path"]
                if execution_result.get("type") == "text" and execution_result.get("value"):
                    st.session_state.current_analysis_artifacts["executed_text_output"] = execution_result.get("value")

                llm_memory_output_for_code_exec = ""
                if execution_result["type"] == "error":
                    current_assistant_response_message[
                        "content"] += f"\n⚠️ **Execution Error:**\n```\n{execution_result['message']}\n```"
                    llm_memory_output_for_code_exec = f"Execution Error: {execution_result['message'][:100]}..."  # Log snippet
                else:
                    current_assistant_response_message["content"] += "\n✅ **Code Executed Successfully!**"
                    # Pass the entire execution_result to the message for the UI to handle display
                    current_assistant_response_message["executed_result"] = execution_result

                    # Add file save paths to the message content for user visibility
                    if execution_result.get("data_path"):
                        current_assistant_response_message[
                            "content"] += f"\n💾 Data from analysis saved to: `{os.path.abspath(execution_result['data_path'])}`"
                    if execution_result.get("plot_path"):
                        current_assistant_response_message[
                            "content"] += f"\n🖼️ Plot image saved to: `{os.path.abspath(execution_result['plot_path'])}`"

                    # Determine memory output based on success type
                    if execution_result["type"] == "table":
                        llm_memory_output_for_code_exec = f"Table result saved: {os.path.basename(execution_result['data_path'])}"
                    elif execution_result["type"] == "plot":
                        llm_memory_output_for_code_exec = f"Plot image saved: {os.path.basename(execution_result['plot_path'])}"
                        if execution_result.get("data_path"):  # If plot also had data CSV
                            llm_memory_output_for_code_exec += f" (with data: {os.path.basename(execution_result['data_path'])})"
                    elif execution_result["type"] == "text":
                        llm_memory_output_for_code_exec = f"Text result: {execution_result['value'][:50]}..."
                    else:
                        llm_memory_output_for_code_exec = "Code executed, unknown result type for memory log."

                # Save context to Langchain memory including generated code and execution outcome
                st.session_state.lc_memory.save_context(
                    {
                        "user_query": user_query + "\n---Generated Code---\n" + generated_code_string + "\n---End Code---"},
                    {"output": llm_memory_output_for_code_exec}
                )

                message_placeholder.markdown(current_assistant_response_message["content"])
                st.session_state.messages.append(current_assistant_response_message)
                st.rerun()
