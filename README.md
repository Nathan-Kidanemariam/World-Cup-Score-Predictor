# World-Cup-Score-Predictor
Machine learning and Poisson probability model for predicting FIFA World Cup match outcomes.

This project uses historical Fifa World Cup data, feature engineering, and trained Random Forest models to estimate the expected goals and create realistic score probabilities.

# Features
* Historical Fifa World Cup data cleaning
* Normalizing country names
* Calculating team strength
* Tracking recent form(last 5 matches) and momentum
* Predicting goals using Machine Learning
* Poisson score probability 
* Predict a single match
* Predict all future tournament matches

# Example Output
France vs Iraq
Expected goals: 3.25 - 0.39
Most likely score: 3-0

France win: 85.9%
Draw: 7.2%
Iraq win: 2.2%

Top scores:
1. 3-0 — 15.0%
2. 2-0 — 13.8%
3. 4-0 — 12.2%

# Project Structure
World-Cup-Score-Predictor/ 
data/ models/ 
src/ 
main.py 
requirements.txt 
README.md

# Installation
Clone the repository:

git clone https://github.com/Nathan-Kidanemariam/World-Cup-Score-Predictor

cd World-Cup-Score-Predictor

Install: 
pip install -r requirements.txt

# Run
Start the predictor:

python main.py

Options:

1. Predict one match
2. Predict all future matches

# Model
This project combines feature engineering, machine learning, and probability all at once
## Feature Engineering
* Attack score
* Defense score
* Momentum
* Team strength
* Recent form
* Goal average

## Machine Learning
* Random Forest Regressor
* Home goal prediction
* Away goal prediction

## Probability
* Poisson distribution
* Win percentage
* Draw percentage
* Loss percentage
* Top scores

# Training Process
* Raw data
* Normalize the names
* clean the world cup matches
* generate team stats
* create training dataset
* train models
* predict matches

# Technologies
* Python
* Pandas
* Scikit-Learn
* Joblib

# Future improvements
* Add Fifa rankings
* Live tournament updating
* tournament simulation



