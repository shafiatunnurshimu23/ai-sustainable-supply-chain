Full Report:
[project_report.docx](https://github.com/user-attachments/files/20968823/project_report.docx)






# 🌍⛓️ AI for Sustainable Supply Chain Management

*A proactive decision-support system to forecast and mitigate carbon emissions in logistics, built with Python, Scikit-learn, and Streamlit.*

---

## 1. Problem Statement

Global supply chains often lack the tools for proactive environmental management, forcing managers to make decisions based on cost and speed while remaining blind to their real-time carbon impact. This reactive approach, relying on annual retrospective reports, is insufficient to meet modern sustainability goals.

This project addresses this gap by designing and building an end-to-end decision-support system that empowers supply chain managers to **forecast the carbon emissions of shipments *before* they occur**. This shifts the paradigm from reactive reporting to proactive, data-driven planning, embedding sustainability directly into the operational workflow.

## 2. The Data: From Raw to Enriched

This analysis was built by integrating two publicly available datasets. The raw data, the processing scripts, and the final enriched dataset are all available in this repository for full transparency.

- **Raw Data:** Sourced from Kaggle, the raw datasets are located in the `/data/raw/` directory.
  - `Supply Chain Logistics Problem.csv`: Contains thousands of individual shipment records.
  - `energy_consumption_by_country.csv`: Provides country-level energy and carbon intensity metrics.

- **Data Processing:** A comprehensive preprocessing and feature engineering pipeline was built using Pandas. This included data cleaning, memory optimization, and engineering critical features like `estimated_distance` and `total_carbon_footprint` by joining the two raw datasets.

- **Final Dataset:** The final, enriched dataset used for modeling is available here:
  - `[final_processed_data.csv](path/to/your/final_data.csv)` <!-- ### TODO: UPDATE THIS LINK ### -->

## 3. Tech Stack

| Category | Technologies |
|---|---|
| **Data Science & ML** | Python, Pandas, NumPy, Scikit-learn (Random Forest), SMOTE (for Imbalanced Data), Joblib |
| **Dashboarding** | Streamlit |
| **Core Tools** | Jupyter Notebook, Git, VS Code |

## 4. Analytical Workflow & Key Findings
### 4.1. Exploratory Data Analysis (EDA)

An initial EDA revealed a heavily right-skewed distribution for shipment weight and a lack of diversity in transportation modes. This insight was critical, as it directly informed our strategy to use **SMOTE** to handle the resulting class imbalance during model training, preventing a bias towards low-emission events.

![figure5_data_distribution](https://github.com/user-attachments/assets/e06facb4-4091-48c5-910b-e1ee51ad166e) <!-- ### TODO: UPDATE THIS LINK ### -->

### 4.2. Predictive Modeling: Identifying High-Risk Shipments

A **Random Forest Classifier** was trained to predict whether a shipment would be 'High-Emission'. The key challenge was the natural class imbalance (only 25% of data was 'High-Emission'). By training on a SMOTE-resampled dataset, the model's performance on the minority class improved significantly.

On the untouched test set, the model achieved a **recall of 0.61** for the critical 'High-Emission' class, successfully identifying 61% of the most carbon-intensive shipments before they occur.

![figure_confusion_matrix_smote](https://github.com/user-attachments/assets/53537050-29d3-4323-a136-d2c0905f4209) <!-- ### TODO: UPDATE THIS LINK ### -->

### 4.3. Feature Importance: Uncovering the Key Drivers

To ensure the model was interpretable, a feature importance analysis was conducted. The results clearly show that `unit_quantity` (shipment size) and `tpt` (transport duration) are the most dominant predictors.

![figure_classification_importances](https://github.com/user-attachments/assets/94a51d4e-7b4a-404f-a7ae-f8a5a722bd89) <!-- ### TODO: UPDATE THIS LINK ### -->
> **Actionable Insight:** The takeaway for a business is clear. The most significant gains in sustainability will come from strategic decisions related to **order consolidation** (to manage `unit_quantity`) and **route optimization** (to manage `tpt`).

## 5. Operationalizing Insights: The Streamlit Dashboard

This project's final output transforms the predictive model from a static script into a dynamic, interactive business tool using Streamlit. This dashboard democratizes the model's insights for non-technical users.

- **Shipment Carbon Forecaster:** Allows managers to perform "what-if" simulations by inputting shipment details to receive an immediate emission risk prediction and a confidence score.
- **Historical Monitoring:** Provides an aggregated overview of historical carbon footprint and a detailed data table for deeper exploration.

![image](https://github.com/user-attachments/assets/8806a5f0-be42-4a84-9db9-bef37eec77e8)
![image](https://github.com/user-attachments/assets/91f033a3-8bfa-4fc1-93bc-9a44aa8bb527)



 <!-- ### TODO: UPDATE THIS LINK ### -->

This tool creates a "human-in-the-loop" system, enabling managers to assess the environmental impact of their decisions *before* they are finalized, closing the gap between data science and practical supply chain operations.

## 6. How to Run This Project

### Prerequisites
- Python 3.8+
- `pip` for package management

### Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shafiatunnurshimu23/ai-sustainable-supply-chain.git
   cd ai-sustainable-supply-chain
2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv ven
   source venv/bin/activate 
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Launch the Streamlit dashboard:**
   ```bash
   python -m streamlit run dashboard.py
Your web browser will automatically open with the running application.

♻️ Work flow for this project given below
```mermaid
graph TD
    subgraph " "
        style A fill:#e6ffed,stroke:#28a745
        style B fill:#e6ffed,stroke:#28a745
        style C fill:#e6ffed,stroke:#28a745
        style D fill:#28a745,stroke:#1e7e34,stroke-width:2px,color:#fff
        A("<b>Raw Data Sources</b><br/>- Supply Chain Shipment Data<br/>- World Energy Consumption Data") --> B("Data Cleaning & Integration");
        B --> C("Feature Engineering<br/>- Estimated Distance<br/>- Transportation Mode<br/>- Manufacturing Carbon Intensity");
        C --> D("<b>Enriched Analysis Dataset</b>");
    end

    subgraph " "
        style E fill:#e6f7ff,stroke:#007bff
        style F fill:#e6f7ff,stroke:#007bff
        style G fill:#007bff,stroke:#0056b3,stroke-width:2px,color:#fff
        D --> E("Predictive Model Training<br/>(Random Forest Classifier)");
        E --> F("Data Leakage Correction & SMOTE");
        F --> G("<b>Key Insights:</b><br/>Feature Importance Analysis<br/>(Identifying Primary Emission Drivers)");
    end

    subgraph " "
        style H fill:#fff3e6,stroke:#fd7e14
        style I fill:#fd7e14,stroke:#c66510,stroke-width:2px,color:#fff
        style J fill:#fff3e6,stroke:#fd7e14
        style K fill:#fff3e6,stroke:#fd7e14
        style L fill:#fd7e14,stroke:#c66510,stroke-width:2px,color:#fff
        D --> H("Interactive Dashboard Development<br/>(Streamlit Prototype)");
        H --> I("<b>Outcome:</b><br/>Real-time Monitoring & Forecasting Platform");
        
        D --> J("Optimization Scenario Modeling<br/>('What-if' Simulation)");
        J --> K("Analysis of Mode Shifting & Sourcing");
        K --> L("<b>Outcome:</b><br/>Framework for Actionable Recommendations");
    end
