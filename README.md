
# 🧠 Multi-Agent Automated Data Analyst

This project is a modular, agent-driven data analysis pipeline built using the [CrewAI](https://docs.crewai.com/) framework. It mimics the behavior of an intelligent data analyst team — each agent with a dedicated role in data handling and analysis — to automatically generate actionable insights from raw CSV files.

---

## 🎯 Objective

Automate the process of:
1. Loading a CSV file
2. Cleaning the data
3. Sorting based on key metrics
4. Generating Exploratory Data Analysis (EDA) visualizations
5. Creating a final, business-readable HTML report

---

## 📦 What's Inside

### Agents Breakdown
| Agent      | Role                    | Task Description |
|------------|-------------------------|------------------|
| **Loader** | Data Ingestion          | Reads CSV into memory and stores it. |
| **Cleaner**| Data Preprocessing      | Removes duplicates, null values, and formats column names. |
| **Sorter** | Data Structuring        | Sorts data based on the first numerical column. |
| **EDA Analyst** | Exploratory Analysis | Creates a correlation heatmap and scatter plots for highly correlated pairs. |
| **Reporter**| Report Generation      | Generates an HTML report with plots, statistics, and summaries. |

---

## 💡 Example Use Case

Let’s say you’re given a new `amazon_delivery.csv` dataset. Rather than manually:
- Checking column names
- Cleaning missing rows
- Plotting correlations
- Writing a report

…you run `main.py`, and the entire process is done automatically — including a beautiful `report.html` with embedded plots. ✅

---

## 🛠️ Technologies Used

- **Python** – Core scripting
- **pandas** – Data manipulation
- **seaborn & matplotlib** – Visualizations
- **CrewAI** – Multi-agent orchestration
- **HTML** – Final business-ready report

---

## 🚀 How to Run

1. Place your CSV file as `amazon_delivery.csv` in the project root.
2. Install dependencies if needed:

```bash
pip install pandas matplotlib seaborn numpy crewai
```

3. Run the project:

```bash
python main.py
```

4. Open the generated `report.html` in any browser.

---

## 📁 Output

- `report.html`: Final report with correlation heatmap, scatter plots, and stats.
- `eda_plots/`: Folder with saved visualizations.
- `amazon_delivery.csv`: Input file used for analysis.

---

## 📌 Notes

- The project uses OpenAI API via `os.environ["OPENAI_API_KEY"]`. Replace the key with your own if needed.
- You can extend this framework with more agents like:
  - Anomaly Detector
  - Feature Engineer
  - AutoML Recommender

---

## 🌟 Why This Project?

It demonstrates how AI automation and modular task assignment can mimic real-world data workflows, making it a perfect showcase for:
- **AI Engineer / Data Engineer interviews**
- **AI system support roles**
- **GitHub portfolios**

---

Made with ❤️ by combining AI agents and classic Python tools.
