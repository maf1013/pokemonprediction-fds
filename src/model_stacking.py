# src/model_stacking.py
from src.utils import *
from src.features import *

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# BASE MODELS
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


#In this section, we define the base models
#(XGBoost, AdaBoost, and Gradient Boosting) that will
#be used within the stacked model (StackingClassifier).


def make_xgb():
    return XGBClassifier(
        n_estimators=1000, #number of trees to train
        learning_rate=0.035, #lower learning rate for more stable training
        max_depth=4, #maximum depth of each tree
        subsample=0.85, # % of samples used per tree
        colsample_bytree=0.85, #% of features used per tree
        reg_lambda=1.0, #L2 regularization to avoid overfitting
        reg_alpha=0.1, #L1 regularization
        random_state=SEED, #for reproductibility
        tree_method="hist", #faster tree-building method
        eval_metric="logloss" #evaluation metric for binary classification
    )

def make_adaboost():
    base = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    return make_pipeline(
        StandardScaler(),
        AdaBoostClassifier(
            estimator=base,
            n_estimators=400, #number of weak classifiers
            learning_rate=0.5, 
            random_state=SEED
        )
    )

def make_gb():
    return GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05, #for boosting
        max_depth=3,
        subsample=0.9, #for regularization
        random_state=SEED
    )

# CROSS VALIDATION

#Here we perform cross-validation for the stacking model. It's trained
#on different folds and we calculate its average accuracy


print("Loading data")
train_raw = load_jsonl(TRAIN_PATH)
test_raw  = load_jsonl(TEST_PATH)
print(f"Train: {len(train_raw)} | Test: {len(test_raw)}")

print("\nCross validation (StackingClassifier: XGB + Ada + GB)...")
from sklearn.model_selection import RepeatedStratifiedKFold

#We use repeated stratified cross-validation (keeps class proportions)
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)

#Indicates who won the battle
y_all = np.array([int(b['player_won']) for b in train_raw])

#Arras to store results
oof_pred_label = np.zeros(len(train_raw), dtype=int)
fold_results = []


#Main training loop for each fold 
#(runs the model on every split of the data)


for fold, (tr_idx, val_idx) in enumerate(skf.split(train_raw, y_all), 1):
    print(f"\nFold {fold}")

    
    # Splitting the data into training and validation sets

    fold_train = [train_raw[i] for i in tr_idx]
    fold_val   = [train_raw[i] for i in val_idx]

    
    # Calculating type effectiveness (feature engineering step to create new fea
    te = learn_type_effectiveness(fold_train, min_battles=15, alpha=1.5, beta=1.5,
                                  min_cap=0.75, max_cap=1.25)

    ## Building the feature dataframes (creates features for training and validation sets)
    tr_df = build_feature_df(fold_train, te)
    va_df = build_feature_df(fold_val, te)
    
    # Remove constant columns
    const_cols = [c for c in tr_df.columns if tr_df[c].nunique(dropna=False) <= 1]
    if const_cols:
        tr_df = tr_df.drop(columns=const_cols) # Drop from training set
        va_df = va_df.drop(columns=[c for c in const_cols if c in va_df.columns])  # Drop from validation set if present

    # Reindex to keep train and validation with the same columns
    va_df = va_df.reindex(columns=tr_df.columns, fill_value=0.0)
    
   # Splitting features and labels (removing ID and target column)
    features = [c for c in tr_df.columns if c not in ['battle_id', 'player_won']]
    X_tr, y_tr = tr_df[features].astype(float).values, tr_df['player_won'].astype(int).values
    X_va, y_va = va_df[features].astype(float).values, va_df['player_won'].astype(int).values


    
    # STACKING CLASSIFIER
     
    #We define the base models and the meta-model (final_estimator)
     
    base_models = [
        ('xgb', make_xgb()),
        ('ada', make_adaboost()),
        ('gb', make_gb())
    ]


     
    # Stacking combines predictions from the base models
    # and uses them as input for a final model (Logistic Regression because of its simplicity, works well in combining predictions)

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=500, C=1.0),
        cv=3, #internal cross-validation for the meta-model
        n_jobs=-1,
        passthrough=False #only uses predictions from base models (not original features)
    )


    # Training the stacking model on the current fold (fits the meta-model using base model predictions)
    stack.fit(X_tr, y_tr)


    #Predicting probabilities on the validation set
    p = stack.predict_proba(X_va)[:, 1]
    pred = (p >= 0.5).astype(int) #convert probabilities to binary predictions

    #Calculating accuracy for this fold
    acc = accuracy_score(y_va, pred)
    oof_pred_label[val_idx] = pred #store out-of-fold predictions

    print(f"Fold {fold} accuracy: {acc:.4f}")
    fold_results.append({"fold": fold, "val_acc": acc})


# FINAL TRAINING + SUBMISSION

#After validating the models, we train the final stacking model with all the training data
#and we generate predictions for the test set.


print("\nTraining final StackingClassifier and generating submission...")


# Compute full type effectiveness (based on battles and smoothing parameters)
te_full = learn_type_effectiveness(train_raw, min_battles=15, alpha=1.5, beta=1.5,
                                   min_cap=0.75, max_cap=1.25)

# Build the final dataset using type effectiveness
train_df = build_feature_df(train_raw, te_full)
test_df  = build_feature_df(test_raw, te_full)

# Remove constant columns 
const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) <= 1]
if const_cols:
    train_df = train_df.drop(columns=const_cols)
    test_df  = test_df.drop(columns=[c for c in const_cols if c in test_df.columns])

# Reindex test set to match training columns
test_df = test_df.reindex(columns=train_df.columns, fill_value=0.0)

# Define input features and target variable
features = [c for c in train_df.columns if c not in ['battle_id', 'player_won']]
X = train_df[features].astype(float).values
y = train_df['player_won'].astype(int).values
X_test = test_df[features].astype(float).values

# Train the final stacking model using all data
stack_final = StackingClassifier(
    estimators=[
        ('xgb', make_xgb()),
        ('ada', make_adaboost()),
        ('gb', make_gb())
    ],
    final_estimator=LogisticRegression(max_iter=500, C=1.0),
    cv=5,
    n_jobs=-1
)

stack_final.fit(X, y)


## Predict probabilities on the test set and convert to binary predictions
p = stack_final.predict_proba(X_test)[:, 1]
pred_test = (p >= 0.5).astype(int)


# Create submission file with predictions
submission = pd.DataFrame({
    'battle_id': test_df['battle_id'].astype(int),
    'player_won': pred_test
})
submission.to_csv('submission_stacking_xgb_ada_gb.csv', index=False)
print(" Saved submission_stacking_xgb_ada_gb.csv")

if __name__ == "__main__":
    print("Running Stacking model...")


