# Credit Card Fraud Detection
Built a multi-model fraud detection pipeline on the PaySim synthetic financial dataset. Engineered balance-change ratio features, handled class imbalance with SMOTE and RandomUnderSampler (applied to training folds only), and trained a five-model ensemble — Decision Tree, Random Forest, Gradient Boosting, MLP Classifier, and a regularized Keras neural network. Used stratified 5-fold cross-validation with fixed random seeds for reproducibility, and evaluated each model across Accuracy, Precision, Recall, F1, and ROC-AUC. Surfaced tree-based feature importances to explain individual predictions and deployed the full pipeline as a Flask + React web app with batch CSV processing, real-time monitoring, and an analyst feedback loop.

## Features
- **File Upload:** Allows users to upload a CSV file containing multiple transactions for batch processing.
- **Real-Time Monitoring:** Displays incoming transactions and their fraud status in real-time.
- **Progress Bar:** Shows upload progress during file processing.
- **Filtering:** Use filters to focus on transactions of particular interest, such as those with high risk or exceeding a specific amount, making it easy to drill down into potential fraudulent activity.
- **Summary Statistics:** Analyze visualized data, including transaction distributions and fraud detection rates, to understand fraud patterns within the bank's transactions.
- **Feedback Loop:** Fraud analysts can mark transactions that have been incorrectly flagged, providing data for future model improvement and making the system more adaptive to new fraud tactics.
  
## Tech Stack
- **Frontend:** React, Axios, HTML, CSS
- **Backend:** Flask, Flask-CORS, Pandas, Joblib
- **Model:** Machine learning model (Decision Trees, MLP Classifier, Ramdom Forest Model, Neural Network, Gradient Boosting) trained for fraud detection using a transaction dataset.
- **Additional Libraries:** Chart.js or Recharts for graphs and visualizations (frontend).
  
## Getting Started
### Prerequisites
- Node.js and npm installed on your machine.
- Python 3 and pip installed.
- Recommended Python libraries: flask, flask-cors, pandas, joblib, sklearn.
  
## Installation
**1. Clone the Repository**
```bash
git clone https://github.com/Danielmark001/MLDA-finfraud.git
cd MLDA-finfraud
```

## Running the Application
**1. Start the Backend (Flask)**
````bash
python app.py
````
The backend will run on `http://localhost:5000.`

**2. Start the Frontend (React)**
In a new terminal window, start the React development server:

````bash
npm start
````
The `frontend` will run on `http://localhost:3000.`

## User Guide
### **1. Upload Transactions for Batch Processing**
Go to the **Upload Transactions** section in the app.
Click the "Browse" button to select a CSV file. A sample template is available for download to format your data correctly.
Click the "Upload" button. A progress bar will indicate the upload status in real-time.
### **2. Real-Time Monitoring**
View incoming transactions in the Real-Time Monitoring section.
Apply filters (amount, risk level, date) to narrow down the displayed transactions.
View summary statistics with interactive charts showing transaction distribution and fraud detection rates.
### **3. Customizing Detection Settings**
Adjust model sensitivity in the settings section (if available).
Use the feedback loop to mark incorrect predictions, helping to improve the model in future versions.

## Sample CSV Template
A sample CSV template (`sample-template.csv`) is provided in the `frontend/public` directory. Users can download and format their transaction data accordingly for batch processing.

## API Endpoints
- **POST /upload:** Uploads a CSV file for batch processing.
- **GET /transactions:** Retrieves the latest transactions for real-time monitoring.
- **POST /feedback:** Allows users to submit feedback on the model's predictions for future improvements.

## Future Enhancements
- **Integration with Bank Systems:** Enable integration with the bank's transaction processing systems for real-time fraud monitoring.
- **Enhanced Authentication:** Add user roles and authentication to control access based on user responsibilities within the bank.
- **Data Storage:** Introduce database support to store transaction history and feedback securely.

## Dataset Source
Axi, E. (2018). Synthetic Financial Datasets For Fraud Detection [Data set]. Kaggle. https://www.kaggle.com/datasets/ealaxi/paysim1/data

Ward, J. (2024). Metaverse Financial Transactions Dataset. https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset
