import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from crewai import Agent, Task, Crew, Process

# Set API Key
os.environ["OPENAI_API_KEY"] = "Enter your API key here"

# Shared data dictionary
shared_data = {}

# Agents
loader = Agent(role="Loader", goal="Load dataset", backstory="Reads and parses CSVs.", verbose=True)
cleaner = Agent(role="Cleaner", goal="Clean data", backstory="Removes nulls and formats headers.", verbose=True)
sorter = Agent(role="Sorter", goal="Sort dataset", backstory="Sorts based on numeric columns.", verbose=True)
eda = Agent(role="EDA Analyst", goal="Visualize and summarize", backstory="Generates EDA summaries.", verbose=True)
reporter = Agent(role="Reporter", goal="Write final report", backstory="Creates business-friendly summaries.", verbose=True)

# Functions (pipeline logic)
def load_csv_task():
    shared_data["df"] = pd.read_csv("amazon_delivery.csv")
    return f"CSV loaded. Shape: {shared_data['df'].shape}"

def clean_data_task():
    df = shared_data["df"]
    df_clean = df.drop_duplicates().dropna()
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")
    shared_data["df_clean"] = df_clean
    return f"Cleaned data. Rows after clean: {len(df_clean)}"

def sort_data_task():
    df = shared_data["df_clean"]
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        return "No numeric columns found to sort."
    df_sorted = df.sort_values(by=numeric_cols[0])
    shared_data["df_sorted"] = df_sorted
    return f"Sorted by column: {numeric_cols[0]}"

def run_eda_task():
    df = shared_data["df_sorted"]
    os.makedirs("eda_plots", exist_ok=True)

    # Save heatmap
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = "eda_plots/heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    shared_data["eda_images"] = [heatmap_path]

    # Top 3 correlated pairs
    corr_pairs = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))
        .unstack()
        .dropna()
        .abs()
        .sort_values(ascending=False)
    )

    plotted = set()
    count = 0
    for (feat1, feat2), value in corr_pairs.items():
        if (feat2, feat1) in plotted or feat1 == feat2:
            continue
        plotted.add((feat1, feat2))
        count += 1
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=feat1, y=feat2, alpha=0.5)
        plt.title(f"{feat1} vs {feat2} (corr={value:.2f})")
        plt.tight_layout()
        img_path = f"eda_plots/{feat1}_vs_{feat2}.png"
        plt.savefig(img_path)
        plt.close()
        shared_data["eda_images"].append(img_path)
        if count >= 3:
            break

    return "EDA complete: Heatmap and scatter plots saved."

def generate_report_task():
    df = shared_data["df_sorted"]
    sample_data_html = df.head(5).to_html(classes='table table-bordered', index=False)
    stats_table_html = df.describe().to_html(classes='table table-striped')

    # Embed all EDA images
    img_html = "".join([f'<img src="{img}" width="700px" style="margin-bottom:30px;"/>' for img in shared_data.get("eda_images", [])])

    report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f8f9fa;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            .table th, .table td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }}
            .table-striped tbody tr:nth-of-type(odd) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>📊 Business Report on Dataset Analysis</h1>
        <h2>1. Dataset Overview</h2>
        <p><strong>Total Rows:</strong> {df.shape[0]}</p>
        <p><strong>Columns:</strong> {', '.join(df.columns)}</p>

        <h3>Sample Data (Top 5 rows)</h3>
        {sample_data_html}

        <h2>2. Data Cleaning</h2>
        <p>Duplicate and null rows have been removed. Data is clean and ready for analysis.</p>

        <h2>3. EDA Visualizations</h2>
        {img_html}

        <h2>4. Summary Statistics</h2>
        {stats_table_html}

        <h2>5. Observations</h2>
        <ul>
            <li>Top correlations are visualized using scatter plots.</li>
            <li>Data distribution and outliers can be inferred from summary statistics and plots.</li>
        </ul>
    </body>
    </html>
    """

    shared_data["html_report"] = report
    return "📄 HTML report with embedded plots generated."

# CrewAI Tasks (just for simulation/logging)
load_task = Task(description="Load CSV file.", expected_output="CSV loaded into shared_data['df']", agent=loader)
clean_task = Task(description="Clean the data.", expected_output="Data cleaned and stored", agent=cleaner)
sort_task = Task(description="Sort data.", expected_output="Sorted dataframe in shared_data", agent=sorter)
eda_task = Task(description="Run EDA.", expected_output="EDA plots saved", agent=eda)
report_task = Task(description="Generate report.", expected_output="Final HTML report created", agent=reporter)

# Crew setup
crew = Crew(
    agents=[loader, cleaner, sorter, eda, reporter],
    tasks=[load_task, clean_task, sort_task, eda_task, report_task],
    process=Process.sequential
)

# Main
if __name__ == "__main__":
    print("API Key set:", os.getenv("OPENAI_API_KEY")[:5] + "..." if os.getenv("OPENAI_API_KEY") else "NOT SET")
    crew.kickoff()

    # Run actual pipeline
    load_csv_task()
    clean_data_task()
    sort_data_task()
    run_eda_task()
    generate_report_task()

    # Save HTML report
    with open("report.html", "w") as f:
        f.write(shared_data.get("html_report", "No report generated."))

    print("✅ 'report.html' generated with heatmap + top 3 scatter plots embedded.")
