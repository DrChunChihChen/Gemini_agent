# simplified_data_agent_streamlit_v3_enhanced.py

import streamlit as st
import pandas as pd
import os
import io  # To read CSV content as string
import json  # For data summary
import datetime  # For timestamping saved files
import matplotlib.pyplot  # Explicit import for placeholder and execution scope
import seaborn  # Explicit import for placeholder and execution scope
import numpy as np  # For numerical operations

# --- Plotly Imports ---
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
LLM_API_KEY = os.environ.get("LLM_API_KEY", "API PLZ")  # Default API Key
# Updated TEMP_DATA_STORAGE to include "AI analysis" subfolder
TEMP_DATA_STORAGE = "temp_data_simplified_agent/AI analysis/"
os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)

AVAILABLE_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro-preview-05-06"]
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
                pass  # Ignore errors if JSON parsing fails

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

            fallback_script = """
# Standard library imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

analysis_result = "Analysis logic executed. If you expected a specific output, please check the generated script."
plot_data_df = None 

# --- AI Generated Code Start ---
# Placeholder: The AI would generate its specific analysis logic here.
# --- AI Generated Code End ---

if 'analysis_result' not in locals() or analysis_result == "Analysis logic executed. If you expected a specific output, please check the generated script.":
    if isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set by the AI's main logic. Displaying df.head() as a default."
        plot_data_df = df.head().copy() 
    else:
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set, and no DataFrame was available."
"""
            if "average sales" in user_query_segment:
                return {"text": "analysis_result = df['sales'].mean()\nplot_data_df = None"}
            elif "plot" in user_query_segment or "visualize" in user_query_segment:
                # Placeholder simulates saving plot to TEMP_DATA_STORAGE and returning only filename
                placeholder_plot_filename = "placeholder_plot.png"
                # Ensure the path for saving uses os.path.join for OS compatibility
                placeholder_full_save_path = os.path.join(TEMP_DATA_STORAGE, placeholder_plot_filename).replace("\\",
                                                                                                                "/")  # Ensure forward slashes for string literal in code

                generated_plot_code = f"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os # Make sure os is imported if os.path.join is used by LLM

fig, ax = plt.subplots()
if not df.empty and len(df.columns) > 0:
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        ax.hist(df[numeric_cols[0]])
        plot_data_df = df[[numeric_cols[0]]].copy()
        # Save the plot to the designated temporary subfolder
        # The placeholder_full_save_path is already correctly formed with TEMP_DATA_STORAGE
        plot_save_path = r'{placeholder_full_save_path}' 
        plt.savefig(plot_save_path)
        plt.close(fig)
        # Return ONLY the filename as analysis_result
        analysis_result = '{placeholder_plot_filename}' 
    else: # Non-numeric data, try a bar plot
        if not df.empty:
            try:
                counts = df.iloc[:, 0].value_counts().head(10) 
                counts.plot(kind='bar', ax=ax)
                plt.title(f'Value Counts for {{df.columns[0]}}')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_data_df = counts.reset_index()
                plot_data_df.columns = [df.columns[0], 'count']
                plot_save_path = r'{placeholder_full_save_path}'
                plt.savefig(plot_save_path)
                plt.close(fig)
                analysis_result = '{placeholder_plot_filename}'
            except Exception as e:
                ax.text(0.5, 0.5, 'Could not generate fallback plot.', ha='center', va='center')
                plot_data_df = pd.DataFrame()
                analysis_result = "Failed to generate fallback plot: " + str(e)
                plt.close(fig)
        else:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
            plot_data_df = pd.DataFrame()
            analysis_result = "No data to plot"
            plt.close(fig)
else:
    ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
    plot_data_df = pd.DataFrame()
    analysis_result = "DataFrame is empty, cannot plot."
    plt.close(fig)
"""
                return {"text": generated_plot_code}
            elif "table" in user_query_segment or "summarize" in user_query_segment:
                return {
                    "text": "analysis_result = df.describe()\nplot_data_df = df.describe().reset_index()"}
            else:
                return {"text": fallback_script}

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
    """
    Retrieves or initializes an LLM instance for the given model name.
    Caches instances in session_state to avoid reinitialization.
    Uses PlaceholderLLM if API key is not properly set.
    """
    if not model_name:
        st.error("No model name provided for LLM initialization.")
        return None
    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}  # Initialize cache if not present

    # If model not in cache, initialize it
    if model_name not in st.session_state.llm_cache:
        if not LLM_API_KEY or LLM_API_KEY == "LLM_API_KEY" or LLM_API_KEY == "YOUR_API_KEY_HERE":  # Check against common placeholder values
            st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
        else:
            try:
                # Set temperature based on whether it's a judge model (more creative/nuanced) or worker (more factual)
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
                st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)  # Fallback to placeholder
    return st.session_state.llm_cache[model_name]


@st.cache_data  # Cache the result of this function for efficiency
def calculate_data_summary(df_input):
    """Calculates a comprehensive summary of the DataFrame."""
    if df_input is None or df_input.empty:
        return None

    df = df_input.copy()  # Work on a copy to avoid modifying the original DataFrame

    # Basic information
    data_summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values_total": int(df.isnull().sum().sum()),  # Total missing values
        "missing_values_per_column": df.isnull().sum().to_dict(),  # Missing values per column
        "descriptive_stats_sample": df.describe(include='all').to_json() if not df.empty else "N/A",
        # Descriptive statistics
        "preview_head": df.head().to_dict(orient='records'),  # First 5 rows
        "preview_tail": df.tail().to_dict(orient='records'),  # Last 5 rows
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),  # List of numeric columns
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        # List of categorical columns
    }
    # Calculate percentage of missing values
    data_summary["missing_values_percentage"] = (data_summary["missing_values_total"] / (
            data_summary["num_rows"] * data_summary["num_columns"])) * 100 if (data_summary["num_rows"] *
                                                                               data_summary[
                                                                                   "num_columns"]) > 0 else 0
    return data_summary


def load_csv_and_get_summary(uploaded_file):
    """Loads a CSV file into a pandas DataFrame and generates its summary."""
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.current_dataframe = df  # Store DataFrame in session state
        st.session_state.data_source_name = uploaded_file.name  # Store filename
        st.session_state.current_analysis_artifacts = {}  # Reset artifacts for new data

        # Calculate and store data summary
        summary_for_state = calculate_data_summary(df.copy())
        if summary_for_state:
            summary_for_state["source_name"] = uploaded_file.name  # Add source name to summary
        st.session_state.data_summary = summary_for_state

        # Reset CDO workflow related session state variables
        st.session_state.cdo_initial_report_text = None
        st.session_state.other_perspectives_text = None
        st.session_state.strategy_text = None
        if "cdo_workflow_stage" in st.session_state:
            del st.session_state.cdo_workflow_stage  # Reset workflow stage
        return True
    except Exception as e:
        st.error(f"Error loading CSV or generating summary: {e}")
        st.session_state.current_dataframe = None
        st.session_state.data_summary = None
        return False


# --- Data Quality Dashboard Functions ---
@st.cache_data  # Cache results for performance
def get_overview_metrics(df):
    """Calculates overview metrics for the data quality dashboard."""
    if df is None or df.empty:
        return 0, 0, 0, 0, 0  # Return zeros if DataFrame is empty
    num_rows = len(df)
    num_cols = len(df.columns)
    missing_values_total = df.isnull().sum().sum()
    total_cells = num_rows * num_cols
    missing_percentage = (missing_values_total / total_cells) * 100 if total_cells > 0 else 0
    numeric_cols_count = len(df.select_dtypes(include=np.number).columns)
    duplicate_rows = df.duplicated().sum()  # Count of duplicate rows
    return num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows


@st.cache_data  # Cache results for performance
def get_column_quality_assessment(df_input):
    """Generates a DataFrame for column-wise quality assessment."""
    if df_input is None or df_input.empty:
        return pd.DataFrame()  # Return empty DataFrame if input is invalid

    df = df_input.copy()
    quality_data = []
    max_cols_to_display = 10  # Limit columns displayed for brevity in dashboard

    for col in df.columns[:max_cols_to_display]:  # Iterate through first 10 columns
        dtype = str(df[col].dtype)
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        unique_values = df[col].nunique()

        # Determine range for numeric/datetime or common values for categorical
        range_common = ""
        if pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].dropna().empty:
                range_common = f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}"
            else:
                range_common = "N/A (all missing)"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):  # Check for datetime types
            if not df[col].dropna().empty:
                range_common = f"Min: {df[col].min()}, Max: {df[col].max()}"
            else:
                range_common = "N/A (all missing)"
        else:  # Categorical columns
            if not df[col].dropna().empty:
                common_vals = df[col].mode().tolist()  # Get most frequent values
                range_common = f"Top: {', '.join(map(str, common_vals[:3]))}"  # Display top 3
                if len(common_vals) > 3:
                    range_common += "..."
            else:
                range_common = "N/A (all missing)"

        # Calculate a simple quality score
        score = 10
        if missing_percent > 50:
            score -= 5
        elif missing_percent > 20:
            score -= 3
        elif missing_percent > 5:
            score -= 1
        if unique_values == 1 and len(df) > 1: score -= 2  # Penalize if only one unique value (constant)
        if unique_values == len(df) and not pd.api.types.is_numeric_dtype(
            df[col]): score -= 1  # Penalize high cardinality non-numeric (potential ID)

        quality_data.append({
            "Column Name": col,
            "Data Type": dtype,
            "Missing %": f"{missing_percent:.2f}%",
            "Unique Values": unique_values,
            "Range / Common Values": range_common,
            "Quality Score ( /10)": max(0, score)  # Score cannot be negative
        })
    return pd.DataFrame(quality_data)


def generate_data_quality_dashboard(df_input):
    """Generates and displays the data quality dashboard using Streamlit elements."""
    if df_input is None or df_input.empty:
        st.warning("No data loaded or DataFrame is empty. Please upload a CSV file.")
        return

    df = df_input.copy()

    st.header("📊 Data Quality Dashboard")
    st.markdown("An overview of your dataset's quality and characteristics.")

    # Display Key Dataset Metrics
    st.subheader("Key Dataset Metrics")
    num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows = get_overview_metrics(df.copy())

    col1, col2, col3, col4, col5 = st.columns(5)  # Use columns for layout
    col1.metric("Total Rows", f"{num_rows:,}")
    col2.metric("Total Columns", f"{num_cols:,}")
    # Conditional coloring for missing values metric
    if missing_percentage > 5:
        col3.metric("Missing Values", f"{missing_percentage:.2f}%", delta_color="inverse",
                    help="Percentage of missing data cells in the entire dataset. Red if > 5%.")
    else:
        col3.metric("Missing Values", f"{missing_percentage:.2f}%",
                    help="Percentage of missing data cells in the entire dataset.")
    col4.metric("Numeric Columns", f"{numeric_cols_count:,}")
    col5.metric("Duplicate Rows", f"{duplicate_rows:,}", help="Number of fully duplicated rows.")

    st.markdown("---")
    # Display Column-wise Quality Assessment
    st.subheader("Column-wise Quality Assessment")
    if len(df.columns) > 10:
        st.caption(
            f"Displaying first 10 columns out of {len(df.columns)}. Full assessment available via report (placeholder).")

    quality_df = get_column_quality_assessment(df.copy())

    if not quality_df.empty:
        # Function to style the quality table (background colors based on values)
        def style_quality_table(df_to_style):
            styled_df = df_to_style.style.apply(
                lambda row: [
                    'background-color: #FFCDD2' if float(str(row["Missing %"]).replace('%', '')) > 20  # High missing %
                    else ('background-color: #FFF9C4' if float(str(row["Missing %"]).replace('%', '')) > 5 else '')
                    # Moderate missing %
                    for _ in row], axis=1, subset=["Missing %"]) \
                .apply(lambda row: ['background-color: #FFEBEE' if row["Quality Score ( /10)"] < 5  # Low quality score
                                    else ('background-color: #FFFDE7' if row["Quality Score ( /10)"] < 7 else '')
                                    # Moderate quality score
                                    for _ in row], axis=1, subset=["Quality Score ( /10)"])
            return styled_df

        st.dataframe(style_quality_table(quality_df), use_container_width=True)
    else:
        st.info("Could not generate column quality assessment table.")

    # Placeholder for full PDF report generation
    if st.button("Generate Full Data Quality PDF Report (Placeholder)", key="dq_pdf_placeholder"):
        st.info("PDF report generation for data quality is not yet implemented.")

    st.markdown("---")
    # Display Numeric Column Distribution
    st.subheader("Numeric Column Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found in the dataset.")
    else:
        selected_numeric_col = st.selectbox("Select Numeric Column for Distribution Analysis:", numeric_cols,
                                            key="dq_numeric_select")
        if selected_numeric_col:
            col_data = df[selected_numeric_col].dropna()  # Drop NA for plotting
            if not col_data.empty:
                # Use Plotly for interactive histogram with box plot marginal
                fig = px.histogram(col_data, x=selected_numeric_col, marginal="box",
                                   title=f"Distribution of {selected_numeric_col}",
                                   opacity=0.75, histnorm='probability density')
                fig.add_trace(  # Add rug plot (invisible markers for layout purposes)
                    go.Scatter(x=col_data, y=[0] * len(col_data), mode='markers', marker=dict(color='rgba(0,0,0,0)'),
                               showlegend=False))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Key Statistics:**")
                stats_cols = st.columns(5)  # Display key stats in columns
                stats_cols[0].metric("Mean", f"{col_data.mean():.2f}")
                stats_cols[1].metric("Median", f"{col_data.median():.2f}")
                stats_cols[2].metric("Std Dev", f"{col_data.std():.2f}")
                stats_cols[3].metric("Min", f"{col_data.min():.2f}")
                stats_cols[4].metric("Max", f"{col_data.max():.2f}")
            else:
                st.info(f"Column '{selected_numeric_col}' contains only missing values.")
    st.markdown("---")

    # Display Categorical Column Distribution
    st.subheader("Categorical Column Distribution")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        st.info("No categorical columns found in the dataset.")
    else:
        selected_categorical_col = st.selectbox("Select Categorical Column for Distribution Analysis:",
                                                categorical_cols, key="dq_categorical_select")
        if selected_categorical_col:
            col_data = df[selected_categorical_col].dropna()  # Drop NA for plotting
            if not col_data.empty:
                value_counts = col_data.value_counts(normalize=True).mul(100).round(
                    2)  # Get value counts as percentages
                count_abs = col_data.value_counts()  # Absolute counts

                # Create labels with both count and percentage for Plotly bar chart
                labels_with_counts_percent = [f"{idx} ({count_abs[idx]}, {val}%)" for idx, val in value_counts.items()]

                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f"Distribution of {selected_categorical_col}",
                             labels={'x': selected_categorical_col, 'y': 'Percentage (%)'},
                             text=[f"{val:.1f}% ({count_abs[idx]})" for idx, val in
                                   value_counts.items()])  # Text on bars

                fig.update_layout(xaxis_title=selected_categorical_col, yaxis_title="Percentage (%)")
                fig.update_traces(textposition='outside')  # Position text outside bars
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Column '{selected_categorical_col}' contains only missing values.")

    st.markdown("---")  # Add a separator
    # --- Numeric Column Correlation Heatmap ---
    st.subheader("Numeric Column Correlation Heatmap")
    numeric_cols_for_corr = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_for_corr) < 2:
        st.info("Not enough numeric columns (at least 2 required) to generate a correlation heatmap.")
    else:
        corr_matrix = df[numeric_cols_for_corr].corr()
        fig_heatmap = px.imshow(corr_matrix,
                                text_auto=True,  # Show correlation values on the heatmap
                                aspect="auto",
                                color_continuous_scale='RdBu_r',  # Red-Blue diverging color scale
                                title="Correlation Heatmap of Numeric Columns")
        fig_heatmap.update_xaxes(side="bottom")  # Ensure x-axis labels are at the bottom
        fig_heatmap.update_layout(
            xaxis_tickangle=-45,  # Angle x-axis labels for better readability
            yaxis_tickangle=0  # Keep y-axis labels horizontal
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


class LocalCodeExecutionEngine:
    """Handles the execution of AI-generated Python code."""

    def execute_code(self, code_string, df_input):
        if df_input is None:
            return {"type": "error", "message": "No data loaded to execute code on."}

        # Prepare a dedicated global and local scope for exec
        exec_globals = globals().copy()  # Start with a copy of current globals
        exec_globals['plt'] = matplotlib.pyplot  # Ensure matplotlib.pyplot is available as plt
        exec_globals['sns'] = seaborn  # Ensure seaborn is available as sns
        exec_globals['pd'] = pd  # Ensure pandas is available as pd
        exec_globals['np'] = np  # Ensure numpy is available as np
        exec_globals['os'] = os  # Ensure os is available if generated code uses os.path.join

        local_scope = {'df': df_input.copy(), 'pd': pd, 'plt': matplotlib.pyplot, 'sns': seaborn, 'np': np, 'os': os}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_df_saved_path = None  # Path for plot-specific data CSV

        # Default message if analysis_result is not set by the script
        default_analysis_result_message = "Code executed, but 'analysis_result' was not explicitly set by the script."
        local_scope['analysis_result'] = default_analysis_result_message
        local_scope['plot_data_df'] = None  # Initialize plot_data_df

        try:
            exec(code_string, exec_globals, local_scope)  # Execute the code string
            analysis_result = local_scope.get('analysis_result')
            plot_data_df = local_scope.get('plot_data_df')

            # Warning if the script didn't set analysis_result
            if analysis_result == default_analysis_result_message:
                st.warning(
                    "The executed script did not explicitly set the 'analysis_result' variable. The output might be incomplete or not as expected.")

            # Handle errors explicitly set by the script
            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                return {"type": "error", "message": analysis_result}

            # Handle plot results: analysis_result is expected to be just the filename
            if isinstance(analysis_result, str) and any(
                    analysis_result.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg"]):
                plot_filename = os.path.basename(analysis_result)  # Sanitize, get filename part

                # The generated code should save the plot to TEMP_DATA_STORAGE/plot_filename
                final_plot_path = os.path.join(TEMP_DATA_STORAGE, plot_filename)

                # Fallback checks if the plot isn't found in the designated subfolder
                if not os.path.exists(final_plot_path):
                    if os.path.exists(analysis_result):  # Check if analysis_result itself is a valid full path
                        final_plot_path = analysis_result
                    else:  # If still not found, report error
                        return {"type": "error",
                                "message": f"Plot file '{plot_filename}' not found. Expected at '{os.path.join(TEMP_DATA_STORAGE, plot_filename)}' or as a direct valid path. `analysis_result` was: '{analysis_result}'. Ensure the generated code saves the plot to the designated temporary directory."}

                # Save plot-specific data if plot_data_df is valid
                if isinstance(plot_data_df, pd.DataFrame) and not plot_data_df.empty:
                    plot_data_filename = f"plot_data_for_{os.path.splitext(plot_filename)[0]}_{timestamp}.csv"
                    plot_data_df_saved_path = os.path.join(TEMP_DATA_STORAGE, plot_data_filename)
                    plot_data_df.to_csv(plot_data_df_saved_path, index=False)
                    st.info(f"Plot-specific data saved to: {plot_data_df_saved_path}")
                elif plot_data_df is not None:  # plot_data_df was set but invalid
                    st.warning(
                        "`plot_data_df` was set by the script but is not a valid or non-empty DataFrame. Not saving associated data for the plot.")
                return {"type": "plot", "plot_path": final_plot_path, "data_path": plot_data_df_saved_path}

            # Handle table results (DataFrame or Series)
            elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                analysis_result = analysis_result.to_frame() if isinstance(analysis_result,
                                                                           pd.Series) else analysis_result
                if analysis_result.empty: return {"type": "text", "value": "The analysis resulted in an empty table."}
                # Save table result to CSV
                saved_csv_path = os.path.join(TEMP_DATA_STORAGE, f"table_result_{timestamp}.csv")
                analysis_result.to_csv(saved_csv_path, index=False)
                return {"type": "table", "data_path": saved_csv_path}
            # Handle other text/numeric results
            else:
                return {"type": "text", "value": str(analysis_result)}
        except Exception as e:
            import traceback  # For detailed error traceback
            error_message_for_user = f"Error during code execution: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            # If error occurred, try to capture what analysis_result was, or default to error
            current_analysis_res = local_scope.get('analysis_result', default_analysis_result_message)
            if current_analysis_res is None or (isinstance(current_analysis_res,
                                                           pd.DataFrame) and current_analysis_res.empty):  # if it's None or an empty DF after error
                local_scope['analysis_result'] = f"Execution Error: {str(e)}"

            return {"type": "error", "message": error_message_for_user,
                    "final_analysis_result_value": local_scope[
                        'analysis_result']}  # Include the final value of analysis_result if possible


code_executor = LocalCodeExecutionEngine()


def export_analysis_to_pdf(artifacts, output_filename="analysis_report.pdf"):
    """Exports the analysis artifacts (CDO report, plot, data, report, critique) to a PDF file."""
    pdf_path = os.path.join(TEMP_DATA_STORAGE, output_filename)  # Save PDF to the "AI analysis" subfolder
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()  # Get default ReportLab styles
    story = []  # List to hold all PDF elements

    # Report Title
    story.append(Paragraph("Comprehensive Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))  # Add space

    # 1. Analysis Goal (User Query)
    story.append(Paragraph("1. Analysis Goal (User Query)", styles['h2']))
    analysis_goal = artifacts.get("original_user_query", "Not specified.")
    story.append(Paragraph(analysis_goal, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # 2. CDO's Initial Data Description & Quality Assessment
    story.append(Paragraph("2. CDO's Initial Data Description & Quality Assessment", styles['h2']))
    cdo_report_text = st.session_state.get("cdo_initial_report_text", "CDO initial report not available.")
    cdo_report_text_cleaned = cdo_report_text.replace("**", "")  # Remove markdown bolding
    for para_text in cdo_report_text_cleaned.split('\n'):  # Add each paragraph
        if para_text.strip().startswith("- "):  # Handle bullet points
            story.append(Paragraph(para_text, styles['Bullet'], bulletText='-'))
        elif para_text.strip():
            story.append(Paragraph(para_text, styles['Normal']))
        else:
            story.append(Spacer(1, 0.1 * inch))  # Small spacer for empty lines
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())  # New page

    # 3. Generated Plot
    story.append(Paragraph("3. Generated Plot", styles['h2']))
    plot_image_path = artifacts.get("plot_image_path")
    if plot_image_path and os.path.exists(plot_image_path):
        try:
            img = Image(plot_image_path, width=6 * inch, height=4 * inch)  # Embed image
            img.hAlign = 'CENTER'  # Center image
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Error embedding plot: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Plot image not available or path incorrect.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # 4. Plot Data (or Executed Data Table)
    story.append(Paragraph("4. Plot Data (or Executed Data Table)", styles['h2']))
    plot_data_csv_path = artifacts.get("executed_data_path")  # Path to the CSV (plot data or table data)
    if plot_data_csv_path and os.path.exists(plot_data_csv_path) and plot_data_csv_path.endswith(".csv"):
        try:
            df_plot = pd.read_csv(plot_data_csv_path)
            data_for_table = [df_plot.columns.to_list()] + df_plot.values.tolist()  # Convert DataFrame to list of lists
            if len(data_for_table) > 1:  # Check if data exists beyond headers
                max_rows_in_pdf = 30  # Limit rows in PDF for brevity
                if len(data_for_table) > max_rows_in_pdf:
                    data_for_table = data_for_table[:max_rows_in_pdf]
                    story.append(
                        Paragraph(f"(Showing first {max_rows_in_pdf - 1} data rows of the CSV)", styles['Italic']))

                # Create and style the table
                table = Table(data_for_table, repeatRows=1)  # repeatRows=1 repeats header on new page
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header background
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align all cells
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header font
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Header padding
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Body background
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)  # Grid lines
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
    story.append(PageBreak())  # New page

    # 5. Generated Textual Report (Specific Analysis)
    story.append(Paragraph("5. Generated Textual Report (Specific Analysis)", styles['h2']))
    report_text_path = artifacts.get("generated_report_path")
    if report_text_path and os.path.exists(report_text_path):
        try:
            with open(report_text_path, 'r', encoding='utf-8') as f:
                report_text_content = f.read()
            report_text_content_cleaned = report_text_content.replace("**", "")  # Remove markdown bolding
            for para_text in report_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;",
                                       styles['Normal']))  # Use &nbsp; for empty lines
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
            critique_text_content_cleaned = critique_text_content.replace("**", "")  # Remove markdown bolding
            for para_text in critique_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading critique text file: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Critique text file not available.", styles['Normal']))

    # Build the PDF
    try:
        doc.build(story)
        return pdf_path
    except Exception as e:
        st.error(f"Failed to build PDF: {e}")
        return None


# --- Streamlit App UI ---
st.set_page_config(page_title="AI CSV Analyst v3.1 (CDO Workflow + DQ Dashboard)", layout="wide")  # Set page config
st.title("🤖 AI CSV Analyst v3.1")
st.caption(
    "Upload CSV, review Data Quality Dashboard, explore data, then optionally run CDO Workflow for AI-driven analysis.")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state: st.session_state.messages = [
    {"role": "assistant",
     "content": "Hello! Select models, upload CSV to view Data Quality Dashboard and start analysis."}]
if "current_dataframe" not in st.session_state: st.session_state.current_dataframe = None
if "data_summary" not in st.session_state: st.session_state.data_summary = None
if "data_source_name" not in st.session_state: st.session_state.data_source_name = None
if "current_analysis_artifacts" not in st.session_state: st.session_state.current_analysis_artifacts = {}
if "selected_worker_model" not in st.session_state: st.session_state.selected_worker_model = DEFAULT_WORKER_MODEL
if "selected_judge_model" not in st.session_state: st.session_state.selected_judge_model = DEFAULT_JUDGE_MODEL
if "lc_memory" not in st.session_state: st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history",
                                                                                              return_messages=False,
                                                                                              input_key="user_query")
# CDO workflow specific state
if "cdo_initial_report_text" not in st.session_state: st.session_state.cdo_initial_report_text = None
if "other_perspectives_text" not in st.session_state: st.session_state.other_perspectives_text = None
if "strategy_text" not in st.session_state: st.session_state.strategy_text = None
if "cdo_workflow_stage" not in st.session_state: st.session_state.cdo_workflow_stage = None  # Tracks current stage of CDO workflow

# --- Prompt Templates ---
# Template for CDO's initial data description
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
# Template for gathering perspectives from department heads
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
# Template for CDO to synthesize analysis suggestions
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

# Updated code_generation_prompt_template for plot saving and analysis_result
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
1.  **`analysis_result` MUST BE SET**: The primary result of your analysis (e.g., a calculated value, a DataFrame, a plot filename string, or a descriptive message) MUST be assigned to a variable named `analysis_result`.
2.  **`plot_data_df` for Plots**: If your analysis involves creating a plot:
    a.  Save the plot to a file within the '{TEMP_DATA_STORAGE}' directory. For example, use a path like `os.path.join('{TEMP_DATA_STORAGE}', 'my_plot.png')` or `'{TEMP_DATA_STORAGE}my_plot.png'`. Ensure `import os` is included if using `os.path.join`.
    b.  Set `analysis_result` to *only the filename string* of this saved plot (e.g., 'my_plot.png'). The application will handle the full path.
    c.  Create a pandas DataFrame named `plot_data_df` containing ONLY the data directly visualized in the chart. If the plot uses the entire `df`, then `plot_data_df = df.copy()`. If no specific data subset is plotted, `plot_data_df` can be an empty DataFrame or `None`.
3.  **`plot_data_df` for Non-Plots**: If the analysis does NOT produce a plot (e.g., it's a table or a single value), `plot_data_df` should generally be set to `None`. However, if `analysis_result` is a DataFrame that you also want to make available for reporting (like a summary table), you can set `plot_data_df = analysis_result.copy()`.
4.  **Default `analysis_result`**: If the user's query is very general (e.g., "explore the data") and no specific plot or table is generated, assign a descriptive string to `analysis_result` (e.g., "Data exploration performed. Key statistics logged or printed.") or assign `df.head()` to `analysis_result`.
5.  **Imports**: Ensure all necessary libraries (`matplotlib.pyplot as plt`, `seaborn as sns`, `pandas as pd`, `numpy as np`, `os` if used for paths) are imported within the script.

**Safety Net - Fallback within your generated script:**
Include the following structure in your generated Python code:
```python
# Initialize analysis_result and plot_data_df at the beginning of your script
analysis_result = "Script started, but 'analysis_result' was not yet set by main logic."
plot_data_df = None

# --- Your main analysis code here ---
# (Example: df_summary = df.describe(); analysis_result = df_summary; plot_data_df = df_summary.copy())
# (Example: import os; plot_path = os.path.join('{TEMP_DATA_STORAGE}', 'age_histogram.png'); plt.hist(df['age']); plt.savefig(plot_path); analysis_result = 'age_histogram.png'; plot_data_df = df[['age']].copy())
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

Python code:""".replace("{TEMP_DATA_STORAGE}", TEMP_DATA_STORAGE.replace("\\", "/"))
    # Ensure forward slashes for prompt path
)
# Template for generating textual reports
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
# Template for judging the AI's analysis
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
1. Code Quality: Correctness, efficiency, readability, best practices, bugs? Correct use of `analysis_result` (filename for plots) and `plot_data_df` (especially if plot generated)? Code saves plot to the correct temporary subfolder?
2. Data Analysis: Relevance to query/data? Accurate transformations/calculations? Appropriate methods? `plot_data_df` content match plot?
3. Plot Quality (if `{plot_image_path}` exists): Appropriate type? Well-labeled? Clear?
4. Report Quality: Clear, concise, insightful? Reflects `data_csv_content`? Addresses query? Accessible language?
5. Overall Effectiveness: How well query addressed (score 1-10)? Actionable suggestions for worker AI (esp. `plot_data_df` and plot saving).
Critique:"""
)

# --- Sidebar UI Elements ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    # Dropdown for selecting worker model
    st.session_state.selected_worker_model = st.selectbox("Select Worker Model:", AVAILABLE_MODELS,
                                                          index=AVAILABLE_MODELS.index(
                                                              st.session_state.selected_worker_model))
    # Dropdown for selecting judge model
    st.session_state.selected_judge_model = st.selectbox("Select Judge Model:", AVAILABLE_MODELS,
                                                         index=AVAILABLE_MODELS.index(
                                                             st.session_state.selected_judge_model))

    st.header("📤 Upload CSV")
    # File uploader for CSV
    uploaded_file = st.file_uploader("Select your CSV file:", type="csv", key="csv_uploader")

    # Process uploaded file
    if uploaded_file is not None:
        # Only reload if it's a new file
        if st.session_state.get("data_source_name") != uploaded_file.name:
            with st.spinner("Processing CSV..."):
                if load_csv_and_get_summary(uploaded_file):
                    st.success(f"CSV '{st.session_state.data_source_name}' processed and ready.")
                    st.session_state.messages.append({  # Add message to chat
                        "role": "assistant",
                        "content": f"Processed '{st.session_state.data_source_name}'. View the Data Quality Dashboard or proceed to other tabs."
                    })
                    st.rerun()  # Rerun to update UI with new data context
                else:
                    st.error("Failed to process CSV.")

    # Display info about loaded file and clear button
    if st.session_state.current_dataframe is not None:
        st.subheader("File Loaded:")
        st.write(
            f"**{st.session_state.data_source_name}** ({len(st.session_state.current_dataframe)} rows x {len(st.session_state.current_dataframe.columns)} columns)")
        if st.button("Clear Loaded Data & Chat", key="clear_data_btn"):
            # Keys to reset in session state
            keys_to_reset = ["current_dataframe", "data_summary", "data_source_name", "current_analysis_artifacts",
                             "messages", "lc_memory", "cdo_initial_report_text", "other_perspectives_text",
                             "strategy_text",
                             "cdo_workflow_stage", "trigger_report_generation", "report_target_data_path",
                             "report_target_plot_path", "report_target_query", "trigger_judging"]
            for key in keys_to_reset:
                if key in st.session_state: del st.session_state[key]

            # Re-initialize messages and memory
            st.session_state.messages = [
                {"role": "assistant", "content": "Data and chat reset. Upload a new CSV file."}]
            st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                                  input_key="user_query")
            st.session_state.current_analysis_artifacts = {}

            # Clean up temporary files from the "AI analysis" subfolder
            cleaned_files_count = 0
            if os.path.exists(TEMP_DATA_STORAGE):
                for item in os.listdir(TEMP_DATA_STORAGE):
                    item_path = os.path.join(TEMP_DATA_STORAGE, item)
                    if os.path.isfile(item_path):  # Only remove files
                        try:
                            os.remove(item_path)
                            cleaned_files_count += 1
                        except Exception as e:
                            st.warning(f"Could not remove temp file {item_path}: {e}")
            st.success(f"Data, chat, and {cleaned_files_count} temporary files from '{TEMP_DATA_STORAGE}' cleared.")
            st.rerun()  # Rerun to reflect cleared state

    st.markdown("---")
    # Display selected models and temp storage path
    st.info(
        f"Worker Model: **{st.session_state.selected_worker_model}**\n\nJudge Model: **{st.session_state.selected_judge_model}**")
    st.info(f"Temporary files stored in: `{os.path.abspath(TEMP_DATA_STORAGE)}`")
    st.warning(  # Security warning about exec()
        "⚠️ **Security Note:** This application uses `exec()` to run AI-generated Python code. This is for demonstration purposes ONLY.")

# --- Main Area with Tabs ---
if st.session_state.current_dataframe is not None:  # Only show tabs if data is loaded
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data Quality Dashboard", "🔍 Data Explorer", "👨‍💼 CDO Workflow", "💬 AI Analysis Chat"])

    # Tab 1: Data Quality Dashboard
    with tab1:
        generate_data_quality_dashboard(st.session_state.current_dataframe.copy())  # Pass a copy of the df

    # Tab 2: Data Explorer
    with tab2:
        st.header("🔍 Data Explorer")
        if st.session_state.data_summary:
            with st.expander("View Full Data Summary (JSON)"):
                st.json(st.session_state.data_summary)
        else:
            st.write("No data summary available yet (should be generated on CSV load).")

        with st.expander(f"View DataFrame Head (First 5 rows of {st.session_state.data_source_name})"):
            st.dataframe(st.session_state.current_dataframe.head())

        with st.expander(f"View DataFrame Tail (Last 5 rows of {st.session_state.data_source_name})"):
            st.dataframe(st.session_state.current_dataframe.tail())

    # Tab 3: CDO Workflow
    with tab3:
        st.header("👨‍💼 CDO-led Analysis Workflow")
        st.markdown(
            "Initiate an AI-driven analysis process involving a simulated Chief Data Officer (CDO) and other department heads.")

        if st.button("🚀 Start CDO Analysis Workflow", key="start_cdo_workflow_btn"):
            st.session_state.cdo_workflow_stage = "initial_description"  # Start the workflow
            # Reset previous CDO workflow outputs
            st.session_state.cdo_initial_report_text = None
            st.session_state.other_perspectives_text = None
            st.session_state.strategy_text = None
            st.session_state.messages.append({"role": "assistant",
                                              "content": f"Starting CDO initial data description with **{st.session_state.selected_worker_model}**..."})
            st.session_state.lc_memory.save_context(  # Log to memory
                {"user_query": f"User initiated CDO workflow for {st.session_state.data_source_name}."},
                {"output": "Requesting CDO initial description."})
            st.rerun()  # Rerun to trigger the next stage

        worker_llm = get_llm_instance(st.session_state.selected_worker_model)

        # Stage 1: CDO Initial Description
        if st.session_state.cdo_workflow_stage == "initial_description":
            if not worker_llm or not st.session_state.data_summary:
                st.error("Worker LLM or data summary not available for CDO workflow.")
            else:
                with st.spinner(
                        f"CDO ({st.session_state.selected_worker_model}) is performing initial data description..."):
                    try:
                        memory_context = st.session_state.lc_memory.load_memory_variables({})
                        data_summary_for_prompt = json.dumps(st.session_state.data_summary,
                                                             indent=2) if st.session_state.data_summary else "{}"

                        cdo_desc_prompt_inputs = {
                            "data_summary": data_summary_for_prompt,
                            "chat_history": memory_context.get("chat_history", "")
                        }
                        formatted_cdo_desc_prompt = cdo_initial_data_description_prompt_template.format_prompt(
                            **cdo_desc_prompt_inputs)
                        response_obj = worker_llm.invoke(formatted_cdo_desc_prompt)
                        cdo_report = response_obj.content if hasattr(response_obj, 'content') else response_obj.get(
                            'text', "Error: CDO description failed.")

                        st.session_state.cdo_initial_report_text = cdo_report
                        st.session_state.messages.append({"role": "assistant",
                                                          "content": f"**CDO's Initial Data Description & Quality Assessment (via {st.session_state.selected_worker_model}):**\n\n{st.session_state.cdo_initial_report_text}"})
                        st.session_state.lc_memory.save_context(
                            {"user_query": "System requested CDO initial data description."},
                            {
                                "output": f"CDO provided initial report: {st.session_state.cdo_initial_report_text[:200]}..."})
                        st.session_state.cdo_workflow_stage = "departmental_perspectives"  # Move to next stage
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during CDO initial data description: {e}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"Error in CDO initial description: {e}"})
                        st.session_state.cdo_workflow_stage = None  # Reset stage on error

        # Stage 2: Departmental Perspectives
        if st.session_state.cdo_workflow_stage == "departmental_perspectives" and st.session_state.cdo_initial_report_text:
            with st.spinner(f"Department Heads (via {st.session_state.selected_worker_model}) are discussing..."):
                try:
                    memory_context = st.session_state.lc_memory.load_memory_variables({})
                    data_summary_for_prompt = json.dumps(st.session_state.data_summary,
                                                         indent=2) if st.session_state.data_summary else "{}"

                    perspectives_prompt_inputs = {
                        "data_summary": data_summary_for_prompt,
                        "chat_history": memory_context.get("chat_history", ""),
                        "cdo_initial_report": st.session_state.cdo_initial_report_text
                    }
                    formatted_perspectives_prompt = individual_perspectives_prompt_template.format_prompt(
                        **perspectives_prompt_inputs)
                    response_obj = worker_llm.invoke(formatted_perspectives_prompt)
                    perspectives = response_obj.content if hasattr(response_obj, 'content') else response_obj.get(
                        'text', "Error: Perspectives generation failed.")

                    st.session_state.other_perspectives_text = perspectives
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**Departmental Perspectives (informed by CDO's report, via {st.session_state.selected_worker_model}):**\n\n{st.session_state.other_perspectives_text}"})
                    st.session_state.lc_memory.save_context(
                        {"user_query": "System requested VPs' perspectives after CDO report."},
                        {"output": f"VPs provided perspectives: {st.session_state.other_perspectives_text[:200]}..."})
                    st.session_state.cdo_workflow_stage = "strategy_synthesis"  # Move to next stage
                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting departmental perspectives: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error getting perspectives: {e}"})
                    st.session_state.cdo_workflow_stage = None  # Reset stage on error

        # Stage 3: Strategy Synthesis
        if st.session_state.cdo_workflow_stage == "strategy_synthesis" and st.session_state.other_perspectives_text:
            with st.spinner(
                    f"CDO ({st.session_state.selected_worker_model}) is synthesizing the final analysis strategy..."):
                try:
                    memory_context = st.session_state.lc_memory.load_memory_variables({})
                    data_summary_for_prompt = json.dumps(st.session_state.data_summary,
                                                         indent=2) if st.session_state.data_summary else "{}"

                    synthesis_prompt_inputs = {
                        "data_summary": data_summary_for_prompt,
                        "chat_history": memory_context.get("chat_history", ""),
                        "cdo_initial_report": st.session_state.cdo_initial_report_text,
                        "generated_perspectives_from_others": st.session_state.other_perspectives_text
                    }
                    formatted_synthesis_prompt = synthesize_analysis_suggestions_prompt_template.format_prompt(
                        **synthesis_prompt_inputs)
                    response_obj = worker_llm.invoke(formatted_synthesis_prompt)
                    strategy = response_obj.content if hasattr(response_obj, 'content') else response_obj.get('text',
                                                                                                              "Error: Strategy synthesis failed.")

                    st.session_state.strategy_text = strategy
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**Final 5 Analysis Strategy Suggestions (by CDO - {st.session_state.selected_worker_model}):**\n\n{st.session_state.strategy_text}\n\nGo to the 'AI Analysis Chat' tab to ask for specific analyses based on these suggestions or your own queries."})
                    st.session_state.lc_memory.save_context(
                        {"user_query": "System requested CDO final strategy synthesis."},
                        {"output": f"CDO provided final strategy: {st.session_state.strategy_text[:200]}..."})
                    st.session_state.cdo_workflow_stage = "completed"  # Mark workflow as completed
                    st.success("CDO Workflow Completed! Check chat history and proceed to AI Analysis Chat tab.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error synthesizing final strategy: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error in CDO final strategy synthesis: {e}"})
                    st.session_state.cdo_workflow_stage = None  # Reset stage on error

        # Display CDO workflow outputs in expanders
        if st.session_state.cdo_initial_report_text:
            with st.expander("CDO's Initial Data Description",
                             expanded=st.session_state.cdo_workflow_stage == "initial_description"):
                st.markdown(st.session_state.cdo_initial_report_text)
        if st.session_state.other_perspectives_text:
            with st.expander("Departmental Perspectives",
                             expanded=st.session_state.cdo_workflow_stage == "departmental_perspectives"):
                st.markdown(st.session_state.other_perspectives_text)
        if st.session_state.strategy_text:
            with st.expander("CDO's Final Analysis Strategy Suggestions",
                             expanded=st.session_state.cdo_workflow_stage == "strategy_synthesis" or st.session_state.cdo_workflow_stage == "completed"):
                st.markdown(st.session_state.strategy_text)

    # Tab 4: AI Analysis Chat
    with tab4:
        st.header("💬 AI Analysis Chat")
        st.caption(
            "Interact with the Worker AI to perform analyses, generate reports, and get critiques from the Judge AI.")
        chat_container = st.container()  # Container for chat messages
        with chat_container:
            for i, message in enumerate(st.session_state.messages):  # Display all messages
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Display results and buttons if it's an assistant message with execution results
                    if message["role"] == "assistant" and "executed_result" in message:
                        executed_res = message["executed_result"]
                        res_type = executed_res.get("type")
                        original_query = message.get("original_user_query",
                                                     st.session_state.current_analysis_artifacts.get(
                                                         "original_user_query", "Unknown query"))

                        # Display table results
                        if res_type == "table":
                            try:
                                df_to_display = pd.read_csv(executed_res["data_path"])
                                st.dataframe(df_to_display)
                                # Button to generate report for the table
                                if st.button(f"📊 Generate Report for this Table##{i}",
                                             key=f"report_table_btn_{i}_tab4"):
                                    st.session_state.trigger_report_generation = True
                                    st.session_state.report_target_data_path = executed_res["data_path"]
                                    st.session_state.report_target_plot_path = None
                                    st.session_state.report_target_query = original_query
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error displaying table from {executed_res['data_path']}: {e}")
                        # Display plot results
                        elif res_type == "plot":
                            if os.path.exists(executed_res["plot_path"]):
                                st.image(executed_res["plot_path"])
                                # Button to generate report for plot data (if available)
                                if executed_res.get("data_path") and os.path.exists(executed_res["data_path"]):
                                    if st.button(f"📄 Generate Report for Plot Data##{i}",
                                                 key=f"report_plot_data_btn_{i}_tab4"):
                                        st.session_state.trigger_report_generation = True
                                        st.session_state.report_target_data_path = executed_res["data_path"]
                                        st.session_state.report_target_plot_path = executed_res["plot_path"]
                                        st.session_state.report_target_query = original_query
                                        st.rerun()
                                else:  # If no specific plot data, offer descriptive report for image
                                    st.caption("Note: Specific data table for this plot was not saved or found.")
                                    if st.button(f"📄 Generate Descriptive Report for Plot Image##{i}",
                                                 key=f"report_plot_desc_btn_{i}_tab4"):
                                        st.session_state.trigger_report_generation = True
                                        st.session_state.report_target_data_path = None
                                        st.session_state.report_target_plot_path = executed_res["plot_path"]
                                        st.session_state.report_target_query = original_query
                                        st.rerun()
                            else:
                                st.warning(f"Plot image not found: {executed_res['plot_path']}")
                        # Display text results
                        elif res_type == "text":
                            st.markdown(
                                f"**Execution Output:**\n```\n{executed_res.get('value', 'No textual output.')}\n```")
                        # Display info about generated report
                        elif res_type == "report_generated":
                            if executed_res.get("report_path") and os.path.exists(executed_res["report_path"]):
                                st.markdown(f"_Report saved to: `{os.path.abspath(executed_res['report_path'])}`_")

                        # Button to judge the analysis
                        artifacts_for_judging = st.session_state.get("current_analysis_artifacts", {})
                        can_judge = artifacts_for_judging.get("generated_code") and \
                                    (artifacts_for_judging.get("executed_data_path") or \
                                     artifacts_for_judging.get("plot_image_path") or \
                                     artifacts_for_judging.get("executed_text_output") or \
                                     (res_type == "text" and executed_res.get("value")))
                        if can_judge:
                            if st.button(f"⚖️ Judge this Analysis by {st.session_state.selected_judge_model}##{i}",
                                         key=f"judge_btn_{i}_tab4"):
                                st.session_state.trigger_judging = True
                                st.rerun()

                    # Display critique and PDF export button
                    if message["role"] == "assistant" and "critique_text" in message:
                        with st.expander(f"View Critique by {st.session_state.selected_judge_model}", expanded=True):
                            st.markdown(message["critique_text"])
                        # Button to export full analysis to PDF
                        if st.button(f"📄 Export Full Analysis to PDF##{i}", key=f"pdf_export_btn_{i}_tab4"):
                            with st.spinner("Generating PDF report..."):
                                pdf_file_path = export_analysis_to_pdf(st.session_state.current_analysis_artifacts)
                                if pdf_file_path and os.path.exists(pdf_file_path):
                                    with open(pdf_file_path, "rb") as pdf_file:
                                        st.download_button(
                                            label="Download Analysis PDF",
                                            data=pdf_file,
                                            file_name=os.path.basename(pdf_file_path),
                                            mime="application/pdf",
                                            key=f"download_pdf_{i}_tab4"
                                        )
                                    st.success(f"PDF report generated: {os.path.basename(pdf_file_path)}")
                                else:
                                    st.error("Failed to generate PDF report.")

        # Chat input for user queries
        if user_query := st.chat_input("Ask for analysis (Worker Model will generate and run code)...",
                                       key="user_query_input_tab4"):
            st.session_state.messages.append({"role": "user", "content": user_query})

            if st.session_state.current_dataframe is None or st.session_state.data_summary is None:
                st.warning("Please upload and process a CSV file first (via sidebar).")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I need CSV data. Please upload a file first."})
            else:
                worker_llm_chat = get_llm_instance(st.session_state.selected_worker_model)
                if not worker_llm_chat:
                    st.error(
                        f"Worker model {st.session_state.selected_worker_model} not initialized. Check API key/selection.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"Worker LLM ({st.session_state.selected_worker_model}) unavailable."})
                else:  # If LLM is available, proceed with code generation
                    with st.chat_message("user"):  # Display user query
                        st.markdown(user_query)
                    st.session_state.current_analysis_artifacts = {"original_user_query": user_query}  # Store query
                    st.session_state.trigger_code_generation = True  # Set trigger for code generation logic
                    st.rerun()  # Rerun to activate the trigger

else:  # If no DataFrame is loaded
    st.info("👋 Welcome! Please upload a CSV file using the sidebar to get started.")
    # Display initial messages if any
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Code Generation Logic (Triggered by user query in chat) ---
if st.session_state.get("trigger_code_generation", False):
    st.session_state.trigger_code_generation = False  # Reset trigger
    user_query = st.session_state.messages[-1]["content"]  # Get the last user query

    with st.chat_message("assistant"):  # Display AI's response in chat
        message_placeholder = st.empty()  # Placeholder for dynamic message updates
        generated_code_string = ""

        message_placeholder.markdown(
            f"⏳ **{st.session_state.selected_worker_model}** generating code for: '{user_query}'...")
        with st.spinner(f"{st.session_state.selected_worker_model} generating Python code..."):
            try:
                worker_llm_code_gen = get_llm_instance(st.session_state.selected_worker_model)
                memory_context = st.session_state.lc_memory.load_memory_variables({})
                data_summary_for_prompt = json.dumps(st.session_state.data_summary,
                                                     indent=2) if st.session_state.data_summary else "{}"

                prompt_inputs = {
                    "data_summary": data_summary_for_prompt,
                    "user_query": user_query,
                    "chat_history": memory_context.get("chat_history", "")
                }
                formatted_prompt = code_generation_prompt_template.format_prompt(**prompt_inputs)
                response_obj = worker_llm_code_gen.invoke(formatted_prompt)
                generated_code_string = response_obj.content if hasattr(response_obj, 'content') else response_obj.get(
                    'text', "")

                # Clean up markdown code block delimiters if present
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

        # Execute the generated code if successful
        if generated_code_string:
            current_assistant_response_message = {"role": "assistant", "content": assistant_base_content,
                                                  "original_user_query": user_query}
            with st.spinner("Executing code..."):
                execution_result = code_executor.execute_code(generated_code_string,
                                                              st.session_state.current_dataframe.copy())

                # Store paths to artifacts from execution
                if execution_result.get("data_path"): st.session_state.current_analysis_artifacts[
                    "executed_data_path"] = execution_result["data_path"]
                if execution_result.get("plot_path"): st.session_state.current_analysis_artifacts["plot_image_path"] = \
                execution_result["plot_path"]
                if execution_result.get("type") == "text" and execution_result.get("value"):
                    st.session_state.current_analysis_artifacts["executed_text_output"] = execution_result.get("value")

                llm_memory_output = ""  # For Langchain memory
                if execution_result["type"] == "error":
                    current_assistant_response_message[
                        "content"] += f"\n⚠️ **Execution Error:**\n```\n{execution_result['message']}\n```"
                    # If exec fails, capture the error in analysis_result if it's still the default
                    if str(st.session_state.current_analysis_artifacts.get("executed_text_output", "")).startswith(
                            "Code executed, but 'analysis_result'"):
                        st.session_state.current_analysis_artifacts[
                            "executed_text_output"] = f"Execution Error: {execution_result.get('final_analysis_result_value', 'Unknown error')}"
                    llm_memory_output = f"Exec Error: {execution_result['message'][:100]}..."
                else:  # Successful execution
                    current_assistant_response_message["content"] += "\n✅ **Code Executed Successfully!**"
                    current_assistant_response_message[
                        "executed_result"] = execution_result  # This dict is used by chat display loop

                    # If the result is text and it's the default message, update the artifact to show this.
                    if execution_result.get("type") == "text" and str(execution_result.get("value", "")).startswith(
                            "Code executed, but 'analysis_result'"):
                        st.session_state.current_analysis_artifacts["executed_text_output"] = str(
                            execution_result.get("value", ""))
                    elif execution_result.get("type") == "text":  # Store any other text output
                        st.session_state.current_analysis_artifacts["executed_text_output"] = str(
                            execution_result.get("value", ""))

                    # Add info about saved files to the message content
                    if execution_result.get("data_path"): current_assistant_response_message[
                        "content"] += f"\n💾 Data saved: `{os.path.abspath(execution_result['data_path'])}`"
                    if execution_result.get("plot_path"): current_assistant_response_message[
                        "content"] += f"\n🖼️ Plot saved: `{os.path.abspath(execution_result['plot_path'])}`"
                    if execution_result.get("data_path") and "plot_data_for" in os.path.basename(
                            execution_result.get("data_path", "")):
                        current_assistant_response_message["content"] += " (Plot-specific data also saved)."

                    # Prepare summary for LLM memory
                    if execution_result["type"] == "table":
                        llm_memory_output = f"Table: {os.path.basename(execution_result['data_path'])}"
                    elif execution_result["type"] == "plot":
                        llm_memory_output = f"Plot: {os.path.basename(execution_result['plot_path'])}"
                        if execution_result.get(
                            "data_path"): llm_memory_output += f" (Data: {os.path.basename(execution_result['data_path'])})"
                    elif execution_result["type"] == "text":
                        llm_memory_output = f"Text: {str(execution_result['value'])[:50]}..."
                    else:
                        llm_memory_output = "Code exec, unknown result type."

                st.session_state.lc_memory.save_context(  # Save to Langchain memory
                    {"user_query": f"{user_query}\n---Code---\n{generated_code_string}\n---End Code---"},
                    {"output": llm_memory_output})
                st.session_state.messages.append(current_assistant_response_message)  # Add full message to chat
                message_placeholder.empty()  # Clear the "generating code..." message
                st.rerun()  # Rerun to display new message and results

# --- Report Generation Logic (Triggered by button press) ---
if st.session_state.get("trigger_report_generation", False):
    st.session_state.trigger_report_generation = False  # Reset trigger
    data_path_for_report = st.session_state.get("report_target_data_path")
    plot_path_for_report = st.session_state.get("report_target_plot_path")
    query_that_led_to_data = st.session_state.report_target_query
    worker_llm_report = get_llm_instance(st.session_state.selected_worker_model)

    if not worker_llm_report or not st.session_state.data_summary or (
            not data_path_for_report and not plot_path_for_report):
        st.error("Cannot generate report: LLM, data summary, or target data/plot path missing.")
    else:
        csv_content_for_report = "N/A - Report is likely descriptive of a plot image."
        if data_path_for_report and os.path.exists(data_path_for_report):  # If specific data CSV exists
            try:
                with open(data_path_for_report, 'r', encoding='utf-8') as f:
                    csv_content_for_report = f.read()
            except Exception as e:
                st.error(f"Error reading data file ('{data_path_for_report}') for report: {e}")
                st.rerun()
        elif plot_path_for_report:  # If no data CSV, but plot exists, it's a descriptive report
            st.info("Generating a descriptive report for the plot image as specific data table was not provided.")

        with st.chat_message("assistant"):  # Display report generation in chat
            report_spinner_msg_container = st.empty()
            report_spinner_msg_container.markdown(
                f"📝 **{st.session_state.selected_worker_model}** is generating report for: '{query_that_led_to_data}'...")
            with st.spinner("Generating report..."):
                try:
                    memory_context = st.session_state.lc_memory.load_memory_variables({})
                    data_summary_for_prompt = json.dumps(st.session_state.data_summary,
                                                         indent=2) if st.session_state.data_summary else "{}"

                    report_prompt_inputs = {
                        "table_data_csv": csv_content_for_report,
                        "original_data_summary": data_summary_for_prompt,
                        "user_query_that_led_to_data": query_that_led_to_data,
                        "chat_history": memory_context.get("chat_history", "")
                    }
                    formatted_prompt = report_generation_prompt_template.format_prompt(**report_prompt_inputs)
                    response_obj = worker_llm_report.invoke(formatted_prompt)
                    report_text = response_obj.content if hasattr(response_obj, 'content') else response_obj.get('text',
                                                                                                                 "Error: Report generation failed.")

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query_part = "".join(c if c.isalnum() else "_" for c in query_that_led_to_data[:30])
                    # Save report to the "AI analysis" subfolder
                    filepath = os.path.join(TEMP_DATA_STORAGE, f"report_for_{safe_query_part}_{timestamp}.txt")
                    with open(filepath, "w", encoding='utf-8') as f:
                        f.write(report_text)

                    st.session_state.current_analysis_artifacts["generated_report_path"] = filepath
                    st.session_state.current_analysis_artifacts["report_query"] = query_that_led_to_data
                    st.session_state.messages.append({  # Add report to chat
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
                    report_spinner_msg_container.empty()  # Clear spinner message
                    st.rerun()  # Rerun to display
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error generating report: {e}"})
                    if 'report_spinner_msg_container' in locals() and report_spinner_msg_container: report_spinner_msg_container.empty()
                    st.rerun()
        # Clean up session state vars for report generation
        for key in ["report_target_data_path", "report_target_plot_path", "report_target_query"]:
            if key in st.session_state: del st.session_state[key]

# --- Judging Logic (Triggered by button press) ---
if st.session_state.get("trigger_judging", False):
    st.session_state.trigger_judging = False  # Reset trigger
    artifacts = st.session_state.current_analysis_artifacts
    judge_llm_judge = get_llm_instance(st.session_state.selected_judge_model)
    original_query_for_artifacts = artifacts.get("original_user_query", "Unknown query for artifacts")

    if not judge_llm_judge or not artifacts.get("generated_code"):
        st.error("Judge LLM not available or generated code missing for critique.")
    else:
        try:
            code_content = artifacts.get("generated_code", "No code found.")
            data_content = "No data file produced or found."  # Default if no data artifact
            if artifacts.get("executed_data_path") and os.path.exists(artifacts["executed_data_path"]):
                with open(artifacts["executed_data_path"], 'r', encoding='utf-8') as f:
                    data_content = f.read()
            elif artifacts.get("executed_text_output"):  # If result was text
                data_content = f"Text output: {artifacts.get('executed_text_output')}"

            report_content = "No report generated or found."  # Default if no report artifact
            if artifacts.get("generated_report_path") and os.path.exists(artifacts["generated_report_path"]):
                with open(artifacts["generated_report_path"], 'r', encoding='utf-8') as f:
                    report_content = f.read()
            elif artifacts.get("report_query") and not artifacts.get("generated_report_path"):
                report_content = f"Report expected for '{artifacts.get('report_query')}' but not found."

            plot_image_actual_path = artifacts.get("plot_image_path", "N/A")
            plot_info_for_judge = f"Plot Image: {plot_image_actual_path}."  # Info for the judge prompt
            if plot_image_actual_path == "N/A":
                plot_info_for_judge = "No plot generated."
            elif not os.path.exists(plot_image_actual_path):
                plot_info_for_judge = f"Plot at '{plot_image_actual_path}' not found."
            else:  # Plot exists
                plot_info_for_judge = f"Plot at '{plot_image_actual_path}'. "
                # Check for associated plot data
                if artifacts.get("executed_data_path") and "plot_data_for" in os.path.basename(
                        artifacts.get("executed_data_path", "")):
                    plot_info_for_judge += f"Plot-specific data at '{artifacts.get('executed_data_path')}'."
                else:
                    plot_info_for_judge += "Plot-specific data not found/applicable."

            with st.chat_message("assistant"):  # Display critique generation in chat
                critique_spinner_msg_container = st.empty()
                critique_spinner_msg_container.markdown(
                    f"⚖️ **{st.session_state.selected_judge_model}** is critiquing analysis for: '{original_query_for_artifacts}'...")
                with st.spinner("Generating critique..."):
                    data_summary_for_prompt = json.dumps(st.session_state.data_summary,
                                                         indent=2) if st.session_state.data_summary else "{}"

                    judging_inputs = {
                        "python_code": code_content, "data_csv_content": data_content,
                        "report_text_content": report_content, "original_user_query": original_query_for_artifacts,
                        "data_summary": data_summary_for_prompt,
                        "plot_image_path": plot_image_actual_path, "plot_info": plot_info_for_judge
                    }
                    formatted_judging_prompt = judging_prompt_template.format_prompt(**judging_inputs)
                    critique_response_obj = judge_llm_judge.invoke(formatted_judging_prompt)
                    critique_text = critique_response_obj.content if hasattr(critique_response_obj,
                                                                             'content') else critique_response_obj.get(
                        'text', "Error: Critique generation failed.")

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query_part = "".join(c if c.isalnum() else "_" for c in original_query_for_artifacts[:30])
                    # Save critique to the "AI analysis" subfolder
                    critique_filename = f"critique_on_{safe_query_part}_{timestamp}.txt"
                    critique_filepath = os.path.join(TEMP_DATA_STORAGE, critique_filename)
                    with open(critique_filepath, "w", encoding='utf-8') as f: f.write(critique_text)
                    st.session_state.current_analysis_artifacts["generated_critique_path"] = critique_filepath

                    st.session_state.messages.append({  # Add critique to chat
                        "role": "assistant",
                        "content": f"⚖️ **Critique by {st.session_state.selected_judge_model} for '{original_query_for_artifacts}' (saved to `{os.path.abspath(critique_filepath)}`):**",
                        "critique_text": critique_text  # This will be displayed by chat loop
                    })
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"Requested critique for: '{original_query_for_artifacts}'"},
                        {"output": f"Critique generated: {critique_text[:100]}..."})
                    critique_spinner_msg_container.empty()  # Clear spinner message
                    st.rerun()  # Rerun to display
        except Exception as e:
            st.error(f"Error during critique generation: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error generating critique: {e}"})
            if 'critique_spinner_msg_container' in locals() and critique_spinner_msg_container: critique_spinner_msg_container.empty()
            st.rerun()
