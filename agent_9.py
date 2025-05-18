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
LLM_API_KEY = os.environ.get("LLM_API_KEY", "LLM_API_KEY")  # Use st.secrets for deployment
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
        prompt_str = str(prompt_input)  # prompt_input can be a PromptValue or a string

        # Ensure prompt_input is converted to string for checks
        if hasattr(prompt_input, 'to_string'):
            prompt_str_content = prompt_input.to_string()
        else:
            prompt_str_content = str(prompt_input)

        if "panel of expert department heads" in prompt_str_content:  # New: Individual perspectives prompt
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
        elif "The following individual perspectives and questions have been generated" in prompt_str_content:  # New: Synthesis prompt
            return {"text": """
*Placeholder Synthesized Suggestions (based on prior perspectives from {model_name}):*
1.  **Strategic Financial Review:** Analyze detailed sales trends (CEO) against product profitability metrics (CFO) to identify high-impact areas for growth.
2.  **Operational Tech Upgrade:** Investigate operational bottlenecks (COO) and assess data automation opportunities (CTO) to improve efficiency in those areas.
3.  **Market-Driven Growth:** Link customer segmentation insights (CMO) with overall market share goals (CEO) to refine marketing strategies.
4.  **Cost-Benefit Analysis for Innovation:** Explore cost drivers (CFO) related to potential tech-driven innovations (CTO) for strategic initiatives (CEO).
5.  **Data Quality for Campaign Efficacy:** Evaluate if current data quality and systems (CTO) are sufficient for accurately measuring marketing campaign effectiveness (CMO) and operational performance (COO).
""".format(model_name=self.model_name)}
        elif "Python code:" in prompt_str_content and "User Query:" in prompt_str_content:  # Code generation
            user_query_segment = prompt_str_content.split("User Query:")[1].split("\n")[0].lower()
            if "average sales" in user_query_segment:
                # OBJECTIVE: Ensure plot_data_df is created even for non-plot results if applicable, or None
                return {"text": "analysis_result = df['sales'].mean()\nplot_data_df = None"}
            elif "plot" in user_query_segment:
                # OBJECTIVE: Placeholder LLM adheres to creating plot_data_df for plots.
                return {
                    "text": "import matplotlib.pyplot as plt\nimport pandas as pd\nfig, ax = plt.subplots()\n# Ensure df has at least one column for plotting\nif not df.empty and len(df.columns) > 0:\n    ax.hist(df.iloc[:, 0])\n    plot_data_df = df.iloc[:, [0]].copy() # Example: data for the first column histogram\nelse:\n    ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')\n    plot_data_df = pd.DataFrame() # Empty dataframe if no plot data\nplot_path = f'{TEMP_DATA_STORAGE}placeholder_plot.png'\nplt.savefig(plot_path)\nplt.close(fig)\nanalysis_result = plot_path\n"}
            else:
                # OBJECTIVE: Ensure plot_data_df is created even for non-plot results if applicable, or None
                return {
                    "text": "analysis_result = df.head()\nplot_data_df = df.head().copy()"}  # Or None if df.head() is not the 'data for report'
        elif "Generate a textual report" in prompt_str_content:
            return {
                "text": f"### Placeholder Report ({self.model_name})\nThis is a placeholder report based on the provided data."}
        elif "Critique the following analysis artifacts" in prompt_str_content:  # Judging prompt
            return {"text": f"""
### Placeholder Critique ({self.model_name})
**Overall Assessment:** The worker AI's output seems to be a placeholder.
**Python Code:** Appears to be Python.
**Data:** Data was mentioned.
**Report:** A report was mentioned.
**Suggestions for Worker AI:** Ensure all outputs are comprehensive and `plot_data_df` is correctly generated for plots.
"""}
        else:
            return {
                "text": f"Placeholder response from {self.model_name} for unrecognized prompt: {prompt_str_content[:200]}..."}


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
                # Temperature is typically set lower for tasks requiring factual/code generation
                # and higher for creative/critique tasks.
                # Judge model might benefit from slightly higher temperature for more nuanced critique.
                temperature = 0.7 if st.session_state.get("selected_judge_model", "") == model_name else 0.2
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

        # Generate a comprehensive data summary
        data_summary = {
            "source_name": uploaded_file.name,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values_per_column": df.isnull().sum().to_dict(),
            "descriptive_stats_sample": df.describe(include='all').to_json() if not df.empty else "N/A",
            # Use to_json for better structure
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

        # Prepare the global scope for exec.
        # This ensures that if the LLM-generated code uses 'global plt' (or pd, sns),
        # it can find these modules.
        exec_globals = globals().copy()  # Start with the script's actual globals
        exec_globals['plt'] = matplotlib.pyplot
        exec_globals['sns'] = seaborn
        exec_globals['pd'] = pd
        # 'df' is intentionally kept out of exec_globals and passed via local_scope
        # as it's context-specific data for the script, not a globally imported module.

        # Prepare a dedicated local scope for exec.
        # This scope provides 'df' and also makes plt, pd, sns available directly
        # if the executed code doesn't use the 'global' keyword for them.
        local_scope = {
            'df': df_input.copy(),
            'pd': pd,
            'plt': matplotlib.pyplot,
            'sns': seaborn
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_df_saved_path = None

        try:
            local_scope[
                'analysis_result'] = "Code executed, but 'analysis_result' was not explicitly set by the script."
            local_scope['plot_data_df'] = None

            # Execute the AI-generated Python code string
            # Pass the enriched exec_globals and the specific local_scope
            exec(code_string, exec_globals, local_scope)

            analysis_result = local_scope.get('analysis_result')
            plot_data_df = local_scope.get('plot_data_df')

            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                return {"type": "error", "message": analysis_result}

            if isinstance(analysis_result, str) and analysis_result.endswith((".png", ".jpg", ".jpeg", ".svg")):
                relative_plot_path = os.path.basename(analysis_result)
                potential_paths = [
                    os.path.join(TEMP_DATA_STORAGE, relative_plot_path),
                    analysis_result,
                    relative_plot_path
                ]
                final_plot_path = None
                for p_path in potential_paths:
                    if os.path.exists(p_path):
                        final_plot_path = p_path
                        break

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
                elif plot_data_df is not None:
                    st.warning(
                        "`plot_data_df` was set by the script but is not a valid or non-empty DataFrame. Not saving associated data for the plot.")
                return {"type": "plot", "plot_path": final_plot_path, "data_path": plot_data_df_saved_path}

            elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                if isinstance(analysis_result, pd.Series):
                    analysis_result = analysis_result.to_frame()

                if analysis_result.empty:
                    return {"type": "text", "value": "The analysis resulted in an empty table."}

                saved_csv_filename = f"table_result_{timestamp}.csv"
                saved_csv_path = os.path.join(TEMP_DATA_STORAGE, saved_csv_filename)
                analysis_result.to_csv(saved_csv_path, index=False)
                return {"type": "table", "data_path": saved_csv_path}
            else:
                return {"type": "text", "value": str(analysis_result)}
        except Exception as e:
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

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Select your models and upload a CSV to get started."}]
if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None
if "data_source_name" not in st.session_state:
    st.session_state.data_source_name = None
if "current_analysis_artifacts" not in st.session_state:  # Stores artifacts like code, data paths, plot paths
    st.session_state.current_analysis_artifacts = {}
if "selected_worker_model" not in st.session_state:
    st.session_state.selected_worker_model = DEFAULT_WORKER_MODEL
if "selected_judge_model" not in st.session_state:
    st.session_state.selected_judge_model = DEFAULT_JUDGE_MODEL

# Initialize Langchain memory
if "lc_memory" not in st.session_state:
    st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                          input_key="user_query")

# --- Prompt Templates ---
# These templates guide the LLM's behavior for different tasks.

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

# OBJECTIVE: This prompt template is crucial. It instructs the LLM on how to generate Python code,
# including the requirement to create `plot_data_df` for plots.
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
    a.  Save the plot to a file. A good name is 'analysis_plot.png'. If you use `plt.savefig('analysis_plot.png')`, it will be in the current working directory.
        You can also save it to a specific temporary folder using a path like '{TEMP_DATA_STORAGE}analysis_plot.png'.
    b.  Set `analysis_result` to the string path of the saved plot file (e.g., 'analysis_plot.png' or '{TEMP_DATA_STORAGE}analysis_plot.png').
    c.  **Crucially, create a pandas DataFrame named `plot_data_df` containing ONLY the data points and categories directly visualized in the chart.**
        For example, if plotting 'total sales per year', `plot_data_df` would be a DataFrame with 'Year' and 'Total Sales' columns.
        If the plot uses a subset or aggregation of `df`, `plot_data_df` should reflect that exact subset/aggregation.
        If the plot happens to use the entire `df` as is, then `plot_data_df = df.copy()`.
        If no specific data subset is plotted (e.g. a plot of a mathematical function not directly from df), `plot_data_df` can be an empty DataFrame or None.
3.  If the analysis results in a table (e.g., a filtered DataFrame, a summary table), `analysis_result` should be this DataFrame or Series. In this case, `plot_data_df` should be set to `None` or not defined unless the table itself is also being plotted.
4.  If the analysis results in a single textual or numerical value, `analysis_result` should be this value. `plot_data_df` should be `None`.
5.  Ensure all necessary libraries (like `import matplotlib.pyplot as plt`, `import seaborn as sns`, `import pandas as pd`) are imported within the script if used.
Do not include any explanations or markdown formatting around the code. Output only the raw Python code.

Python code:""".replace("{TEMP_DATA_STORAGE}", TEMP_DATA_STORAGE)  # Inject storage path
)

# OBJECTIVE: This prompt uses `table_data_csv`, which will contain the content of `plot_data_df` (if a plot was made)
# or the content of `analysis_result` (if it was a table).
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

**Your Report Structure and Content Guidelines:**

* **1. Executive Summary (1-2 sentences):**
    * Immediately state the main conclusion or the most significant finding derived *directly* from the provided "Core Data for Your Analysis" in relation to the "User's Specific Query".

* **2. Purpose of this Analysis (1 sentence):**
    * Briefly restate what the user was trying to achieve with their query: "{user_query_that_led_to_data}".

* **3. Key Observations from the Data (Bulleted list, 2-4 points):**
    * Detail the most important patterns, trends, anomalies, or specific data points visible in the "Core Data for YourAnalysis".
    * Quantify your observations where possible (e.g., "Category X showed a 20% increase," "The average value is Y").
   
* **4. Actionable Insights & Interpretations (1-2 distinct insights):**
    * Go beyond just stating facts. What do these observations *mean* in the context of the "Original Dataset Summary" and the "User's Specific Query"?
    * What potential implications or next questions arise from these findings?
    * Offer interpretations that a business user would find valuable. Avoid speculation; base insights firmly on the provided "Core Data for Your Analysis".

* **5. Data Focus & Limitations (Brief note):**
    * Clearly state that your report is based *solely* on the "Core Data for Your Analysis" provided.
    * If `{table_data_csv}` is "N/A", mention that the insights are based on a general understanding of the described plot type rather than specific data points.

**Tone and Style:**
* Professional, authoritative, and clear.
* Use precise language. Avoid vague statements.
* Assume the reader is intelligent but may not be a data expert. Explain concepts simply if necessary.
* Do NOT refer to the data as "the CSV" or "the table data." Instead, talk about "the analysis results," "the findings," or "the data indicates."

**Begin Report:**
---
Report:
"""
)

judging_prompt_template = PromptTemplate(
    input_variables=["python_code", "data_csv_content", "report_text_content", "original_user_query", "data_summary",
                     "plot_image_path", "plot_info"],
    template="""You are an expert data science reviewer and critique AI.
Your task is to meticulously evaluate the analysis artifacts produced by another AI assistant in response to a user query.

Original User Query: "{original_user_query}"
Original Data Summary (of the input dataset): {data_summary}

--- ARTIFACTS FOR REVIEW ---
1.  Generated Python Code:
    ```python
    {python_code}
    ```

2.  Data Produced by Code (if applicable, content in CSV format, or description of text output):
    ```csv
    {data_csv_content}
    ```
    {plot_info}

3.  Generated Report (if applicable):
    ```text
    {report_text_content}
    ```
--- END ARTIFACTS ---

Please provide a structured critique covering the following aspects:
1.  **Code Quality & Correctness:**
    * Does the code correctly implement the logic to address the user query?
    * Is it efficient and readable? Does it follow Python best practices? Any bugs or potential issues?
    * **Crucially, if a plot was generated, did the code also correctly create and assign the `plot_data_df` variable containing the exact data used for that plot?**
    * Does it correctly use `analysis_result` as instructed?

2.  **Data Analysis & Relevance:**
    * Is the analysis performed relevant to the user's query and the nature of the data?
    * Are the transformations and calculations accurate (based on the code)?
    * Is the choice of analysis methods appropriate?
    * If `plot_data_df` was created, does its content accurately represent the data visualized in the plot described by `plot_info`?

3.  **Plot Quality (if a plot was generated and `plot_info` indicates its presence):**
    * Assuming the plot image at `{plot_image_path}` is viewable, is the plot type appropriate for the data and query?
    * Is it well-labeled (title, axes, legends)?
    * Does it effectively convey information? Is it clear or cluttered?

4.  **Report Quality (if a report was generated):**
    * Is the report clear, concise, and insightful?
    * **Does it accurately reflect the data provided in `data_csv_content` (which would be from `plot_data_df` for plots, or `analysis_result` for tables)?**
    * Does it adequately address the user query and provide meaningful interpretations based *only* on the provided data?
    * Is the language accessible?

5.  **Overall Effectiveness & Adherence to Query:**
    * How well did the worker AI, across all artifacts (code, data output, plot, report), address the user's original query?
    * Provide an overall score (1-10, where 10 is excellent).
    * Offer specific, actionable suggestions for the worker AI to improve in future, similar tasks, especially regarding the creation and use of `plot_data_df`.

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
        # Use a unique key for the button to avoid issues with reruns
        if st.button("Process CSV & Get Suggestions", key="process_csv_btn"):
            with st.spinner("Processing CSV..."):
                if load_csv_and_get_summary(uploaded_file):
                    st.success(f"CSV '{st.session_state.data_source_name}' processed.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Processed '{st.session_state.data_source_name}'. Data summary is available. Now, let's get some initial perspectives from different department heads using **{st.session_state.selected_worker_model}**..."
                    })
                    st.session_state.lc_memory.save_context(
                        {
                            "user_query": f"Uploaded {st.session_state.data_source_name} and requested initial perspectives."},
                        {"output": "Processing CSV and preparing for perspective generation."}
                    )
                    # Reset artifacts for this new dataset
                    st.session_state.current_analysis_artifacts = {}

                    worker_llm = get_llm_instance(st.session_state.selected_worker_model)
                    if worker_llm and st.session_state.data_summary:
                        perspectives_text = ""
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

                                perspectives_response_obj = worker_llm.invoke(formatted_perspectives_prompt)
                                if hasattr(perspectives_response_obj, 'content'):
                                    perspectives_text = perspectives_response_obj.content
                                elif isinstance(perspectives_response_obj,
                                                dict) and 'text' in perspectives_response_obj:
                                    perspectives_text = perspectives_response_obj['text']
                                else:
                                    perspectives_text = str(perspectives_response_obj)

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"**Initial Perspectives from Department Heads (via {st.session_state.selected_worker_model}):**\n\n{perspectives_text}\n\nNow, synthesizing these into actionable suggestions..."
                                })
                                st.session_state.lc_memory.save_context(
                                    {"user_query": "System requested individual department perspectives."},
                                    {"output": f"Provided perspectives: {perspectives_text[:200]}..."}
                                )
                            except Exception as e:
                                st.error(f"Error getting individual perspectives: {e}")
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": f"Error getting perspectives: {e}"})
                                perspectives_text = "Error generating perspectives."  # Ensure this is set for the check below

                        if perspectives_text and "Error generating perspectives." not in perspectives_text:
                            with st.spinner(
                                    f"🛠️ {st.session_state.selected_worker_model} is synthesizing actionable suggestions..."):
                                try:
                                    memory_context = st.session_state.lc_memory.load_memory_variables({})
                                    synthesis_prompt_inputs = {
                                        "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                        "chat_history": memory_context.get("chat_history", ""),
                                        "generated_perspectives": perspectives_text
                                    }
                                    formatted_synthesis_prompt = synthesize_analysis_suggestions_prompt_template.format_prompt(
                                        **synthesis_prompt_inputs)

                                    synthesis_response_obj = worker_llm.invoke(formatted_synthesis_prompt)
                                    if hasattr(synthesis_response_obj, 'content'):
                                        strategy_text = synthesis_response_obj.content
                                    elif isinstance(synthesis_response_obj, dict) and 'text' in synthesis_response_obj:
                                        strategy_text = synthesis_response_obj['text']
                                    else:
                                        strategy_text = str(synthesis_response_obj)

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
                            st.warning(
                                "Skipping synthesis of suggestions due to error in generating perspectives or empty perspectives.")
                    else:
                        st.error("Worker LLM or data summary not available for generating suggestions.")
                    st.rerun()  # Rerun to update the main chat interface
                else:
                    st.error("Failed to process CSV.")

    if st.session_state.current_dataframe is not None:
        st.subheader("File Loaded:")
        st.write(
            f"**{st.session_state.data_source_name}** ({len(st.session_state.current_dataframe)} rows x {len(st.session_state.current_dataframe.columns)} columns)")
        with st.expander("View Data Summary & Details"):
            if st.session_state.data_summary:
                st.json(st.session_state.data_summary)  # Display as JSON for readability
            else:
                st.write("No data summary available.")
        with st.expander("View DataFrame Head (First 5 rows)"):
            st.dataframe(st.session_state.current_dataframe.head())

        if st.button("Clear Loaded Data & Chat", key="clear_data_btn"):
            keys_to_reset = [
                "current_dataframe", "data_summary", "data_source_name",
                "current_analysis_artifacts", "messages", "lc_memory",
                "trigger_report_generation", "report_target_data_path",
                "report_target_plot_path", "report_target_query", "trigger_judging"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]

            # Re-initialize essential states
            st.session_state.messages = [
                {"role": "assistant", "content": "Data and chat reset. Upload a new CSV file."}]
            st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                                  input_key="user_query")
            st.session_state.current_analysis_artifacts = {}

            # Clean up temporary files
            cleaned_files_count = 0
            for item in os.listdir(TEMP_DATA_STORAGE):
                if item.endswith((".png", ".csv", ".txt", ".jpg", ".jpeg", ".svg")):
                    try:
                        os.remove(os.path.join(TEMP_DATA_STORAGE, item))
                        cleaned_files_count += 1
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {item}: {e}")
            st.success(f"Data, chat, and {cleaned_files_count} temporary files cleared.")
            st.rerun()

    st.markdown("---")
    st.info(
        f"Worker Model: **{st.session_state.selected_worker_model}**\n\nJudge Model: **{st.session_state.selected_judge_model}**")
    st.info(f"Temporary files are stored in: `{os.path.abspath(TEMP_DATA_STORAGE)}`")
    st.warning(
        "⚠️ **Security Note:** This application uses `exec()` to run AI-generated Python code. This is for demonstration purposes ONLY and can be a security risk if used with untrusted code or in a production environment.")

# --- Main Chat Interface ---
# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display execution results (plots, tables) and buttons for reports/judging
        if message["role"] == "assistant" and "executed_result" in message:
            executed_res = message["executed_result"]
            res_type = executed_res.get("type")
            # Ensure original_user_query is available for context
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
                        st.session_state.report_target_plot_path = None  # No plot for table report
                        st.session_state.report_target_query = original_query
                        st.rerun()
                except Exception as e:
                    st.error(f"Error displaying table from {executed_res['data_path']}: {e}")

            elif res_type == "plot":
                if os.path.exists(executed_res["plot_path"]):
                    st.image(executed_res["plot_path"])
                    # OBJECTIVE: Button to generate report using the specific plot data CSV
                    if executed_res.get("data_path") and os.path.exists(executed_res["data_path"]):
                        if st.button(f"📄 Generate Report for Plot Data##{i}", key=f"report_plot_data_btn_{i}"):
                            st.session_state.trigger_report_generation = True
                            st.session_state.report_target_data_path = executed_res[
                                "data_path"]  # Path to the plot's specific data CSV
                            st.session_state.report_target_plot_path = executed_res[
                                "plot_path"]  # Plot path for context
                            st.session_state.report_target_query = original_query
                            st.rerun()
                    else:  # Fallback if plot_data_df.csv was not created/found
                        st.caption(
                            "Note: Specific data table for this plot was not saved or found. Report will be descriptive of the image if generated.")
                        if st.button(f"📄 Generate Descriptive Report for Plot Image##{i}",
                                     key=f"report_plot_desc_btn_{i}"):
                            st.session_state.trigger_report_generation = True
                            st.session_state.report_target_data_path = None  # No specific data CSV
                            st.session_state.report_target_plot_path = executed_res["plot_path"]
                            st.session_state.report_target_query = original_query
                            st.rerun()
                else:
                    st.warning(f"Plot image not found at path: {executed_res['plot_path']}")

            elif res_type == "text":
                st.markdown(f"**Execution Output:**\n```\n{executed_res.get('value', 'No textual output.')}\n```")

            elif res_type == "report_generated":  # Indicates a report text was generated and saved
                if executed_res.get("report_path") and os.path.exists(executed_res["report_path"]):
                    st.markdown(f"_Report text saved to: `{os.path.abspath(executed_res['report_path'])}`_")

            # Check if there are artifacts available to be judged
            artifacts_for_judging = st.session_state.get("current_analysis_artifacts", {})
            # Judge if code was generated AND (data OR plot OR text output was produced)
            can_judge = artifacts_for_judging.get("generated_code") and \
                        (artifacts_for_judging.get("executed_data_path") or \
                         artifacts_for_judging.get("plot_image_path") or \
                         artifacts_for_judging.get("executed_text_output") or \
                         (res_type == "text" and executed_res.get("value")))  # also consider direct text output

            if can_judge:
                if st.button(f"⚖️ Judge this Analysis by {st.session_state.selected_judge_model}##{i}",
                             key=f"judge_btn_{i}"):
                    st.session_state.trigger_judging = True
                    st.rerun()

        # Display critique if available in the message
        if message["role"] == "assistant" and "critique_text" in message:
            with st.expander(f"View Critique by {st.session_state.selected_judge_model}", expanded=True):
                st.markdown(message["critique_text"])

# --- Report Generation Logic ---
if st.session_state.get("trigger_report_generation", False):
    st.session_state.trigger_report_generation = False  # Reset trigger
    data_path_for_report = st.session_state.get("report_target_data_path")
    plot_path_for_report = st.session_state.get("report_target_plot_path")  # For context, if report is about a plot
    query_that_led_to_data = st.session_state.report_target_query

    worker_llm = get_llm_instance(st.session_state.selected_worker_model)

    if not worker_llm or not st.session_state.data_summary:
        st.error("Cannot generate report: LLM or original data summary missing.")
        st.rerun()  # Use rerun to refresh UI state
    if not data_path_for_report and not plot_path_for_report:  # Must have at least plot path for descriptive or data path for data-driven
        st.error("Cannot generate report: Target data or plot path missing.")
        st.rerun()

    # OBJECTIVE: Load the content of the specific data CSV (e.g., plot_data_df.csv) for the report.
    csv_content_for_report = "N/A - Report is likely descriptive of a plot image."
    if data_path_for_report and os.path.exists(data_path_for_report):
        try:
            with open(data_path_for_report, 'r', encoding='utf-8') as f:
                csv_content_for_report = f.read()
        except Exception as e:
            st.error(f"Error reading data file ('{data_path_for_report}') for report: {e}")
            st.rerun()
    elif plot_path_for_report:  # If no data_path, but there's a plot_path, it's a descriptive report
        st.info("Generating a descriptive report for the plot image as specific data table was not provided.")

    with st.chat_message("assistant"):  # Show report generation in chat
        report_spinner_msg_container = st.empty()
        report_spinner_msg_container.markdown(
            f"📝 **{st.session_state.selected_worker_model}** is generating a report for query: '{query_that_led_to_data}'...")

        with st.spinner("Generating report... please wait."):  # Spinner for the LLM call
            try:
                memory_context = st.session_state.lc_memory.load_memory_variables({})
                report_prompt_inputs = {
                    "table_data_csv": csv_content_for_report,  # This is the key data for the report
                    "original_data_summary": json.dumps(st.session_state.data_summary, indent=2),
                    "user_query_that_led_to_data": query_that_led_to_data,
                    "chat_history": memory_context.get("chat_history", "")
                }

                formatted_prompt = report_generation_prompt_template.format_prompt(**report_prompt_inputs)

                response_obj = worker_llm.invoke(formatted_prompt)
                if hasattr(response_obj, 'content'):
                    report_text = response_obj.content
                elif isinstance(response_obj, dict) and 'text' in response_obj:  # PlaceholderLLM
                    report_text = response_obj['text']
                else:
                    report_text = str(response_obj)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize filename from query
                safe_query_part = "".join(c if c.isalnum() else "_" for c in query_that_led_to_data[:30])
                base_filename = f"report_for_{safe_query_part}_{timestamp}.txt"
                filepath = os.path.join(TEMP_DATA_STORAGE, base_filename)
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(report_text)

                # Update current analysis artifacts with report info
                st.session_state.current_analysis_artifacts["generated_report_path"] = filepath
                st.session_state.current_analysis_artifacts[
                    "report_query"] = query_that_led_to_data  # Query that led to this report

                # Append report to chat messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"📊 **Report by {st.session_state.selected_worker_model} for query '{query_that_led_to_data}':**\n\n{report_text}",
                    "original_user_query": query_that_led_to_data,  # Keep track of original query for this message
                    "executed_result": {  # Store info about the generated report itself
                        "type": "report_generated",
                        "report_path": filepath,
                        "data_source_path": data_path_for_report if data_path_for_report else "N/A (Plot Description)",
                        "plot_source_path": plot_path_for_report if plot_path_for_report else "N/A"
                    }
                })
                st.session_state.lc_memory.save_context(
                    {"user_query": f"Requested report for analysis related to: '{query_that_led_to_data}'"},
                    {"output": f"Report generated and saved to {filepath}: {report_text[:100]}..."}
                )
                report_spinner_msg_container.empty()  # Clear spinner message
                st.rerun()  # Refresh UI
            except Exception as e:
                st.error(f"Error generating report: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error generating report: {e}"})
                if report_spinner_msg_container: report_spinner_msg_container.empty()
                st.rerun()

    # Clean up session state vars for report generation to prevent re-triggering without button press
    if "report_target_data_path" in st.session_state: del st.session_state.report_target_data_path
    if "report_target_plot_path" in st.session_state: del st.session_state.report_target_plot_path
    if "report_target_query" in st.session_state: del st.session_state.report_target_query

# --- Judging Logic ---
if st.session_state.get("trigger_judging", False):
    st.session_state.trigger_judging = False  # Reset trigger
    artifacts = st.session_state.current_analysis_artifacts
    judge_llm = get_llm_instance(st.session_state.selected_judge_model)

    original_query_for_artifacts = artifacts.get("original_user_query", "Unknown original query for these artifacts")

    if not judge_llm:
        st.error("Judge LLM not available. Cannot perform critique.")
        st.rerun()
    if not artifacts.get("generated_code"):
        st.error("Cannot perform critique: Generated Python code artifact is missing.")
        st.rerun()

    try:
        code_content = artifacts.get("generated_code", "No code was generated or found.")

        # Prepare data_csv_content for the judge
        data_content = "No specific data file (CSV) was produced by the code, or it was not found."
        # OBJECTIVE: Ensure judge gets the plot-specific data if it exists
        if artifacts.get("executed_data_path") and os.path.exists(
                artifacts["executed_data_path"]):  # This could be plot_data_df.csv or table_result.csv
            with open(artifacts["executed_data_path"], 'r', encoding='utf-8') as f:
                data_content = f.read()
        elif artifacts.get("executed_text_output"):  # If code produced text output instead of a file
            data_content = f"Text output from code: {artifacts.get('executed_text_output')}"

        report_content = "No report was generated or found for this analysis cycle."
        if artifacts.get("generated_report_path") and os.path.exists(artifacts["generated_report_path"]):
            with open(artifacts["generated_report_path"], 'r',
                      encoding='utf-8') as f:  # FIX: Corrected encoding from utf-f8 to utf-8
                report_content = f.read()
        elif artifacts.get("report_query") and not artifacts.get(
                "generated_report_path"):  # A report was expected but not found
            report_content = f"A report was expected for the query '{artifacts.get('report_query')}' but its file was not found or not generated."

        plot_image_actual_path = artifacts.get("plot_image_path", "N/A")
        plot_info_for_judge = f"Plot Image Path: {plot_image_actual_path}."
        if plot_image_actual_path == "N/A":
            plot_info_for_judge = "No plot image was generated or specified for this analysis."
        elif not os.path.exists(plot_image_actual_path):
            plot_info_for_judge = f"A plot was reportedly generated at '{plot_image_actual_path}', but the file is not found. Judge based on code if it intended to create one, and whether `plot_data_df` was appropriately generated."
        else:  # Plot exists
            plot_info_for_judge = f"Plot image is expected at '{plot_image_actual_path}'. "
            if artifacts.get("executed_data_path") and "plot_data_for" in os.path.basename(
                    artifacts.get("executed_data_path")):
                plot_info_for_judge += f"Associated plot-specific data CSV is expected at '{artifacts.get('executed_data_path')}'."
            else:
                plot_info_for_judge += "Associated plot-specific data CSV was not found or not applicable."

        with st.chat_message("assistant"):  # Show critique generation in chat
            critique_spinner_msg_container = st.empty()
            critique_spinner_msg_container.markdown(
                f"⚖️ **{st.session_state.selected_judge_model}** is critiquing the analysis for query: '{original_query_for_artifacts}'...")
            with st.spinner("Generating critique... please wait."):  # Spinner for the LLM call
                judging_inputs = {
                    "python_code": code_content,
                    "data_csv_content": data_content,  # This will be plot_data_df's content for plots
                    "report_text_content": report_content,
                    "original_user_query": original_query_for_artifacts,
                    "data_summary": json.dumps(st.session_state.data_summary,
                                               indent=2) if st.session_state.data_summary else "N/A",
                    "plot_image_path": plot_image_actual_path,  # Actual path for judge's reference
                    "plot_info": plot_info_for_judge  # Contextual info about the plot for the judge
                }
                formatted_judging_prompt = judging_prompt_template.format_prompt(**judging_inputs)

                critique_response_obj = judge_llm.invoke(formatted_judging_prompt)
                if hasattr(critique_response_obj, 'content'):
                    critique_text = critique_response_obj.content
                elif isinstance(critique_response_obj, dict) and 'text' in critique_response_obj:  # PlaceholderLLM
                    critique_text = critique_response_obj['text']
                else:
                    critique_text = str(critique_response_obj)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query_part = "".join(c if c.isalnum() else "_" for c in original_query_for_artifacts[:30])
                critique_filename = f"critique_on_{safe_query_part}_{timestamp}.txt"
                critique_filepath = os.path.join(TEMP_DATA_STORAGE, critique_filename)
                with open(critique_filepath, "w", encoding='utf-8') as f:
                    f.write(critique_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚖️ **Critique by {st.session_state.selected_judge_model} for analysis of '{original_query_for_artifacts}' (saved to `{os.path.abspath(critique_filepath)}`):**",
                    "critique_text": critique_text  # This will be displayed by the chat loop
                })
                st.session_state.lc_memory.save_context(
                    {
                        "user_query": f"Requested critique for analysis related to query: '{original_query_for_artifacts}'"},
                    {"output": f"Critique generated and saved to {critique_filepath}: {critique_text[:100]}..."}
                )
                critique_spinner_msg_container.empty()  # Clear spinner message
                st.rerun()  # Refresh UI
    except Exception as e:
        st.error(f"Error during critique generation: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error generating critique: {e}"})
        # Ensure spinner message container is cleared if it exists
        if 'critique_spinner_msg_container' in locals() and critique_spinner_msg_container is not None:
            critique_spinner_msg_container.empty()
        st.rerun()

# --- Chat Input and Code Generation/Execution Logic ---
if user_query := st.chat_input("Ask for analysis (Worker Model will generate and run code)...", key="user_query_input"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

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
        message_placeholder = st.empty()  # For dynamic updates of this message
        generated_code_string = ""
        assistant_base_content = ""  # Will hold the generated code part of the message

        # Reset/initialize analysis artifacts for this new query
        st.session_state.current_analysis_artifacts = {
            "original_user_query": user_query,
            "generated_code": None,
            "executed_data_path": None,  # Path to table_result.csv or plot_data_for_PLOT.csv
            "plot_image_path": None,
            "executed_text_output": None,
            "generated_report_path": None,
            "report_query": None
        }

        message_placeholder.markdown(
            f"⏳ **{st.session_state.selected_worker_model}** is thinking and generating Python code for your query: '{user_query}'...")
        with st.spinner(f"{st.session_state.selected_worker_model} is generating Python code..."):
            try:
                memory_context = st.session_state.lc_memory.load_memory_variables({})
                prompt_inputs = {
                    "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                    "user_query": user_query,
                    "chat_history": memory_context.get("chat_history", "")
                }
                formatted_prompt = code_generation_prompt_template.format_prompt(**prompt_inputs)

                response_obj = worker_llm.invoke(formatted_prompt)
                if hasattr(response_obj, 'content'):
                    generated_code_string = response_obj.content
                elif isinstance(response_obj, dict) and 'text' in response_obj:  # PlaceholderLLM
                    generated_code_string = response_obj['text']
                else:
                    generated_code_string = str(response_obj)

                # Clean up potential markdown code block delimiters from LLM output
                if generated_code_string.startswith("```python"):
                    generated_code_string = generated_code_string[len("```python"):].strip()
                elif generated_code_string.startswith("```"):  # More general case
                    generated_code_string = generated_code_string[len("```"):].strip()
                if generated_code_string.endswith("```"):
                    generated_code_string = generated_code_string[:-len("```")].strip()

                st.session_state.current_analysis_artifacts["generated_code"] = generated_code_string
                assistant_base_content = f"🔍 **Generated Python Code by {st.session_state.selected_worker_model} for query '{user_query}':**\n```python\n{generated_code_string}\n```\n"
                message_placeholder.markdown(assistant_base_content + "\n⏳ Now executing this code locally...")

            except Exception as e:
                error_msg = f"Error generating code with {st.session_state.selected_worker_model}: {e}"
                st.error(error_msg)  # Show error in main UI
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.lc_memory.save_context({"user_query": user_query},
                                                        {"output": f"Code Generation Error: {e}"})
                st.rerun()  # Refresh UI

        if generated_code_string:  # Proceed if code generation was successful
            current_assistant_response_message = {
                "role": "assistant",
                "content": assistant_base_content,  # Start with the generated code
                "original_user_query": user_query  # Store the query that led to this
            }

            with st.spinner("Executing generated code locally... Please wait."):
                execution_result = code_executor.execute_code(generated_code_string, st.session_state.current_dataframe)

                # Store paths to artifacts from execution
                # OBJECTIVE: `execution_result["data_path"]` will be the path to `plot_data_df.csv` if a plot was made and data saved.
                if execution_result.get("data_path"):  # This could be table data or plot-specific data
                    st.session_state.current_analysis_artifacts["executed_data_path"] = execution_result["data_path"]
                if execution_result.get("plot_path"):
                    st.session_state.current_analysis_artifacts["plot_image_path"] = execution_result["plot_path"]
                if execution_result.get("type") == "text" and execution_result.get("value"):
                    st.session_state.current_analysis_artifacts["executed_text_output"] = execution_result.get("value")

                llm_memory_output_for_code_exec = ""  # For logging to conversation memory
                if execution_result["type"] == "error":
                    current_assistant_response_message[
                        "content"] += f"\n⚠️ **Execution Error:**\n```\n{execution_result['message']}\n```"
                    llm_memory_output_for_code_exec = f"Execution Error: {execution_result['message'][:100]}..."
                else:
                    current_assistant_response_message["content"] += "\n✅ **Code Executed Successfully!**"
                    # This `executed_result` dict will be used by the chat display loop to show data/plot and buttons
                    current_assistant_response_message["executed_result"] = execution_result

                    # Add info about saved files to the message content
                    if execution_result.get("data_path"):  # General data path (table or plot-specific)
                        current_assistant_response_message[
                            "content"] += f"\n💾 Data from analysis saved to: `{os.path.abspath(execution_result['data_path'])}`"
                    if execution_result.get("plot_path"):
                        current_assistant_response_message[
                            "content"] += f"\n🖼️ Plot image saved to: `{os.path.abspath(execution_result['plot_path'])}`"
                        # Specifically mention if plot-specific data was also saved
                        if execution_result.get("data_path") and "plot_data_for" in os.path.basename(
                                execution_result.get("data_path")):
                            current_assistant_response_message["content"] += f" (Plot-specific data also saved)."

                    # Prepare summary for LLM memory
                    if execution_result["type"] == "table":
                        llm_memory_output_for_code_exec = f"Table result saved: {os.path.basename(execution_result['data_path'])}"
                    elif execution_result["type"] == "plot":
                        llm_memory_output_for_code_exec = f"Plot image saved: {os.path.basename(execution_result['plot_path'])}"
                        if execution_result.get("data_path"):  # If plot_data_df.csv was saved
                            llm_memory_output_for_code_exec += f" (with specific data for report: {os.path.basename(execution_result['data_path'])})"
                    elif execution_result["type"] == "text":
                        llm_memory_output_for_code_exec = f"Text result: {execution_result['value'][:50]}..."
                    else:
                        llm_memory_output_for_code_exec = "Code executed, unknown result type for memory log."

                # Save context to Langchain memory
                st.session_state.lc_memory.save_context(
                    {
                        "user_query": user_query + "\n---Generated Code---\n" + generated_code_string + "\n---End Code---"},
                    {"output": llm_memory_output_for_code_exec}
                )

                message_placeholder.markdown(
                    current_assistant_response_message["content"])  # Update the placeholder with full content
                st.session_state.messages.append(current_assistant_response_message)  # Add to chat history
                st.rerun()  # Refresh UI to show new messages and results properly
