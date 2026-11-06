# .github

Financial-Distress Early-Warning Assistant
Project Overview

This tool helps bank employees find customers who may be in financial trouble. It scores every customer on seven simple rules (card usage, late payments, balance, overdraft, minimum payment, balance drop and loan burden), explains each flag, and uses a neural network built with PyTorch to give a second opinion. A chatbot lets staff ask questions in plain English.

It replaces slow, manual account checks with fast screening of all 5,000 customers, so the bank can offer support earlier. Flags are only prompts for an employee to review, not automatic decisions.

Built by Team CC for Business Analysis 3.2, Vaal University of Technology.

Project Workflow
Data: Comprehensive_Banking_Database.csv (5,000 customers, 2023). The rules flag customers as NO FLAG, EARLY WARNING or FINANCIAL DISTRESS, and about 20% are flagged. Age, gender and city are never used for scoring.
Training: the neural network learns from the rules' flags, using an 80/20 train/test split and 5-fold cross-validation. It is compared with logistic regression, random forest and gradient boosting.
Evaluation: tested on 1,000 customers it has never seen, using precision, recall, F1-score, ROC-AUC, a confusion matrix, and ROC and precision-recall curves. Results are saved in the reports folder.
Deployment

The assistant runs as a chatbot in your browser (web_app.py) or in the terminal (bank_distress_bot.py). The browser version only runs on your own computer. Every question and review is saved in an audit log.

Installation Guide
Clone the repository
   git clone https://github.com/<your-username>/<repository-name>.git
   cd <repository-name>
Install the requirements
   pip install -r requirements.txt
Train the model and run the tests
   cd src
   python train_model.py
   python test_distress.py
Start the chatbot
   python web_app.py --open
How to Use
Start the browser chat with python web_app.py --open.
Click a suggested question or type your own, for example:
Find customers at risk
Check customer 1234
Why was customer 1234 flagged?
To record a review, add a note and your employee ID: resolve flag for 1234 by E1234: spoke to customer
Press Clear to start a new conversation.

The assistant shows the customer's status, score and the reasons behind it.

Other Scripts

Run these from the src folder:

python analysis.py for data charts and a fairness check
python time_series.py for weekly trends and forecasts
python deep_learning_extras.py tune for model tuning
python deep_learning_extras.py autoencoder for anomaly detection
python nlp_intent.py to train the question classifier
python predict_model.py --top 20 for the highest-risk customers
