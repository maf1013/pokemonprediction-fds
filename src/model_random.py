from src.utils import *
from src.features import *

import numpy as np
import pandas as pd


def run_random():
    print("Running Random model...")

    
    # Load data
    
    train_raw = load_jsonl(TRAIN_PATH)
    test_raw = load_jsonl(TEST_PATH)

    # Compute type effectiveness using ALL train data
    te_full = learn_type_effectiveness(
        train_raw, 
        min_battles=15, alpha=1.5, beta=1.5,
        min_cap=0.75, max_cap=1.25
    )

    # Build feature dataframes
    train_df = build_feature_df(train_raw, te_full)
    test_df  = build_feature_df(test_raw, te_full)

   
    # Remove constant columns
   
    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) <= 1]
    if const_cols:
        print(f"Removing constant columns: {const_cols}")
        train_df = train_df.drop(columns=const_cols)
        test_df  = test_df.drop(columns=[c for c in const_cols if c in test_df.columns])

   #Select all feature columns except 'battle_id' and the target column
    features = [c for c in train_df.columns if c not in ['battle_id','player_won']]
    X = train_df[features].astype(float)
    y = train_df['player_won'].astype(int)
    #Prepare the test feature matriz using the same columns
    X_test = test_df[features].astype(float)

   #Train a Random Forest model with cross-validation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    #Create a startified k-fold splitter with 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    #Arrays to store predictions OOF (out-of-fold( for training and averaged predictions for test
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}")

        #Split data into training and validation sets for this fold
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        #Initialize Random Forest model with chosen hyperparameters
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=SEED,
            n_jobs=-1
        )

        model.fit(X_tr, y_tr)
        #Predict on validation set and store result
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

        print(f"Fold {fold+1} accuracy: {accuracy_score(y_val, preds):.4f}")

    print("\nOOF Accuracy:", accuracy_score(y, oof_preds))

    #we create the final Random Forest model
    final_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        criterion='entropy',
        random_state=SEED,
        n_jobs=-1
    )

    # we train the model with the entire trining dataset
    final_model.fit(X, y)

    # we make predictions on the test set
    final_pred = final_model.predict(X_test)

    # Create submission file with predictions
    submission = pd.DataFrame({
        'battle_id': test_df['battle_id'].astype(int),
        'player_won': final_pred.astype(int)
    })

    submission.to_csv("submission_random.csv", index=False)
    print("\nCSV generated: submission_random.csv")



if __name__ == "__main__":
    run_random()
