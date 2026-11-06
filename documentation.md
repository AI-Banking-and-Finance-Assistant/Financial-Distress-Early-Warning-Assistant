# **Documentation Aspects:**



#### **AI Solution**



The Financial-Distress Early-Warning Assistant by team CC is an intelligent early warning assistant based on artificial intelligence technology. The system scores each bank customer against a weighted criteria list of rules of account health and categorizes them into one of the three statuses: NO FLAG, EARLY WARNING or FINANCIAL DISTRESS and provides easy access to the information via a chatbot to bank employees. It demonstrates that intelligent automation can be used instead of traditional manual work of reviewing customer accounts in the finance industry. Therefore, this solution falls under the theme of “AI Solutions for Industries.” The implementation of MMG AI in the banking industry helps with better customer service, minimizes unnecessary losses and supports the agenda of Fourth Industrial Revolution (4IR).



### **Business objectives**



#### **Business objectives**



The goal of the Financial Distress Early Warning Assistant is to create a rule-based scoring
system that can detect customers who are already experiencing financial distress through
their accounts. This assists the bankers in intervening early and allows them to help such
customers. It also gives a justification for each warning raised.


**The goals of The Financial-Distress Early-Warning Assistant are as follows:**

* Develop a reliable algorithm that will mark those customers with money problems,
with each marking justified.
* Aid the analyst in prioritizing customers with warnings (multiple warnings, high
scores).
* Save on review time and money by analysing all 5,000 customers in milliseconds.
* Make the information more accessible through a chatbot in the terminal and in the
browser.
* Enhance customer wellbeing through the recommendation of appropriate assistance
for every warning.



#### **Business success criteria**



* Each flag can be explained: the chatbot’s "why" and "rules" responses give the
exact criteria used to flag a status.
* 5,000 customers are loaded and scored within approximately 0.2 seconds (target:
under 60 seconds).
* Response time for the chat question is approximately 0.01 seconds or faster
(timed).
* The bank’s anomaly flagger (6.0% of customers) is presented to the user along with
the flags.
* Bank employees can work with the assistant without any tech skills using
recommended questions and natural language.
* The system accounts for missing or corrupt data in an entire batch.



#### **Business background**



Team CC is a technology company in the financial service sector working with retail banks
to ensure customers' financial well-being. Financial distress, like overdrawn accounts, full
credit cards, and non-payments, is typically identified too late and manually by an employee
checking one account at a time. This results in unnecessary losses to the bank and further
debts for the client. The industry requires technology which screens all customers equally
and provides a rationale behind the decision. The Financial-Distress Early-Warning
Assistant fits into this need by employing consistent guidelines across the board for all
clients and providing a way to query the output generated.



#### **Requirements**



* Data: Comprehensive\_Banking\_Database.csv (5,000 customers, 40 features, 2023 year of transactions)
* Python 3 along with packages pandas, NumPy, Matplotlib and scikit-learn; SQLite (default) for audit log
* Thresholds for rules in one config dictionary
* Assessment via accuracy, balanced accuracy, precision, recall, F1-score and ROCAUC for diagnostic models

&#x20;

#### **Constraints**



* Dataset Limitation: Only 5,000 customers with their single most recent transactions from 2023.
* Rule Coverage Limitation: Only seven criteria are scored; any other factors are ignored by the rules engine.
* Threshold Update: Periodic update of the thresholds is required since both the customer's behaviour and policies may change at the bank's end.
* Does Not Replace Human Judgment: The system assists analysts but does not decide on the customers; the web page explicitly states it.
* Temporal Information Limitation: Trends are detected through transaction dates and snapshots of information that are taken every time when the new data file is uploaded.
* Computationally Efficient: It works on a standard laptop without GPU; web interface is bound to localhost only due to personal information display.
* Language Limitation: English only. Requests that the chatbot cannot recognize will be reformulated using the optional Gemini model.
* Compliance with Regulations: It must abide by POPIA and FICA; age, gender, and city are never considered in scores



#### &#x20;**Risks**



|Risk|Impact|Probability|Mitigation Strategy|
|-|-|-|-|
|Data Quality|High - one bad row <br />used to crash <br />scoring for all <br />customers|Medium|Missing dates, phone numbers and <br />type text now use safe fallbacks; each <br />row is scored in its own try/except, so a <br />bad record is skipped and logged <br />(tested on a deliberately corrupted file)|
|Wrong or <br />Missed Flags|<br />High - customers <br />wrongly flagged or <br />distress missed|Medium|Every flag shows its signals and points; <br />employees review flags; resolutions <br />are logged and distress cases need <br />supervisor approval; no automatic <br />action is taken|
|Label <br />Leakage|Medium - <br />misleadingly high <br />diagnostic accuracy|High|Charts print a leakage note whenever a <br />model predicts the rule engine's own <br />label; such results are never used for <br />decisions|
|Demographic <br />Bias|Medium - unequal <br />flag rates across <br />groups|Medium|Age, gender and city are never inputs; <br />a demographics chart audits flag rates <br />by group, with small groups faded|
|Privacy <br />Breach|High - exposure of <br />customer personal <br />data|Medium|Browser server bound to localhost; <br />every query and review written to an <br />audit log; POPIA/FICA compliance <br />required|
|Over-reliance|Medium - staff <br />treating flags as <br />decisions|Medium|Interface states flags are for employee <br />review only; reasons are shown; <br />human sign-off required|



#### **Tools and techniques** 



**Programming Environment:** 

* Language: Python 3.10+ 
* Version Control: Git \& GitHub (for project management) 

**Core Libraries:** 

* pandas \& NumPy (Data Handling) 
* scikit-learn (diagnostic machine learning and text-term counting) 
* Matplotlib (Visualization) 
* SQLite / sqlite3 (audit log, snapshots and flag reviews) and http.server (browser interface), both built in 
* Optional: Google Gemini API (rephrases unrecognised questions only) 

**Machine Learning Technique:** 

* Algorithm: Weighted rule-based scoring (main); Random Forest and Logistic Regression as diagnostics only 
* Class balancing: class\_weight="balanced" in diagnostic models 
* Data preprocessing: StandardScaler in case of Logistic Regression, and the rule engine operates on the original account fields directly. 
* Performance evaluation: Accuracy, classification report, confusion matrices, ROC, and feature importance.



### **Problem definition** 



#### **What is the problem?**

#### &#x20;

One of the significant issues in the field of banking is the delayed identification of a customer's financial difficulty. The existence of overdrafts, maxed out credit cards, delays in payments, and loan problems comes to attention only when there is a loss of money, since nobody reviews all the clients. According to the statistics of the AI, 20.36% of customers have enough indications of their problems to attract the attention of specialists, but at present, nobody detects them and identifies the reasons. The traditional approach to the review of customers relies on the identification of difficulties in one account by a bank teller or an analyst, but this process is rather slow, inefficient, and hard to detect in busy branches. 



#### **How relevant is it to the theme, and how beneficial will it be in solving the problem?**

&#x20;

This project is highly relevant to the theme of "AI Solutions for Industries" since it leverages intelligent automation on the finance industry which is a very important part of society. It transforms the tedious, manual and subjective process into an automated, explainable and scalable process, thereby fulfilling the promise of 4IR which seeks to leverage data and automation to enhance industry processes. The result is a win-win situation for both the bank and the customer, since the former will be saving money and avoiding risks while the latter will be assisted before defaulting. 

