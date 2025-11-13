# Pokémon Battle Prediction Project

## Authors
- Rebeca Ojer Aransáez (rebeca.2246423@studenti.uniroma1.it)
- María Acevedo Fernández (maria.2245832@studenti.uniroma1.it)

This project aims to predict which player wins a Pokémon battle using machine learning models.  
To achieve this, we transform the raw battle data into a set of numerical features and train several models capable of generating prediction files compatible with Kaggle.

Three main models are developed and trained:

- **Random Forest**  
- **Voting Classifier** (XGBoost + AdaBoost)  
- **Stacking Classifier** (XGBoost + AdaBoost + Gradient Boosting)

Each model produces its own prediction CSV file.

---

## Project Overview

The work includes:

- Feature engineering applied to battle data  
- Implementation of multiple machine learning models  
- A final Kaggle notebook that documents the entire process

---

## Included Models

The `src/` directory contains the implementations of the three models:

- **Random Forest**  
- **Voting Classifier** (XGBoost + AdaBoost)  
- **Stacking Classifier** (XGBoost + AdaBoost + Gradient Boosting)

Each script generates a prediction file (`submission_*.csv`) ready to be submitted to Kaggle.

---

## Final Notebook

The notebook located in `/notebook/` includes:

- Feature engineering  
- Detailed function descriptions  
- Model justification and rationale  
- Results  
- Submission generation
