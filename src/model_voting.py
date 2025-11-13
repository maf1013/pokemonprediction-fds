# src/model_voting.py

from src.utils import *
from src.features import *

# Imports necesarios
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold



"""
We define two base classifiers that will  be combined using a VotingClassifier:
XGBoost – a powerful non-parametric model based on gradient-boosted decision trees.
AdaBoost – a classic boosting algorithm that uses  decision trees, offering simplicity and robustness.
"""

"""
XGBoost with moderate hyperparameters to reduce overfitting:
low max_depth limits model complexity, while a small learning_rate
ensures stable learning.
"""

def make_xgb():
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.06,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        reg_alpha=0.1,
        random_state=SEED,
        tree_method="hist",
        eval_metric="logloss"
    )


"""
AdaBoost with a very simple base estimator (max_depth=2): each tree focuses
on correcting previous errors,
resulting in a model that’s resilient to noise.
"""
def make_adaboost():
    base = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    return AdaBoostClassifier(
        estimator=base,          
        n_estimators=400,
        learning_rate=0.6,
        random_state=SEED
    )


#CROSS VALIDATION
def run_voting():
    print("Loading data")
    train_raw = load_jsonl(TRAIN_PATH)
    test_raw  = load_jsonl(TEST_PATH)
    print(f"Train: {len(train_raw)} | Test: {len(test_raw)}")


    """
    We split the training set into 5 folds to estimate
    the model’s average
    performance while preventing data leakage.
    """

    print("\nCross validation (VotingClassifier: XGB + AdaBoost)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_all = np.array([int(b['player_won']) for b in train_raw])

    oof_pred_label = np.zeros(len(train_raw), dtype=int)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_raw, y_all), 1):
        print(f"\nFold {fold}")

        #We split the training and validation data for this fold to evaluate the model without mixing the sets.
        fold_train = [train_raw[i] for i in tr_idx]
        fold_val   = [train_raw[i] for i in val_idx]

        # We compute the type effectiveness table using only the training data from this fold to avoid leaking information from the validation set.
        te = learn_type_effectiveness(fold_train, min_battles=15, alpha=1.5, beta=1.5,
                                    min_cap=0.75, max_cap=1.25)

        # We generate the features for both the training and validation sets.
        tr_df = build_feature_df(fold_train, te)
        va_df = build_feature_df(fold_val, te)

        # We remove constant columns and make sure the features are properly aligned.
        const_cols = [c for c in tr_df.columns if tr_df[c].nunique(dropna=False) <= 1]
        if const_cols:
            tr_df = tr_df.drop(columns=const_cols)
            va_df = va_df.drop(columns=[c for c in const_cols if c in va_df.columns])

        # We defensively align the columns to make sure everything matches across datasets.
        va_df = va_df.reindex(columns=tr_df.columns, fill_value=0.0)
    
        # We select the predictor variables and target, then extract training and validation arrays for model input.
        features = [c for c in tr_df.columns if c not in ['battle_id', 'player_won']]
        X_tr, y_tr = tr_df[features].astype(float).values, tr_df['player_won'].astype(int).values
        X_va, y_va = va_df[features].astype(float).values, va_df['player_won'].astype(int).values

    
    #COMBINED MODELS
        """
        We build an ensemble using VotingClassifier with soft
        voting to combine XGBoost’s strength and AdaBoost’s reliability.
        """
    
        voter = VotingClassifier(
            estimators=[("xgb", make_xgb()), ("ada", make_adaboost())],
            voting="soft",
            weights=[0.5, 0.5]
        )

        # We train the full ensemble model using the training data from this fold.
        voter.fit(X_tr, y_tr)

        # We make predictions on the validation set to check how well the model performs.
        p = voter.predict_proba(X_va)[:, 1]
        pred = (p >= 0.5).astype(int)
        acc = accuracy_score(y_va, pred)
        oof_pred_label[val_idx] = pred

        print(f"Fold {fold} accuracy: {acc:.4f}")
        fold_results.append({"fold": fold, "val_acc": acc})



    #FINAL TRAINING + SUBMISSION

    print("\nTraining the final model and generating the submission...")

    # We update the type effectiveness table using the full training set to make it more accurate.
    te_full = learn_type_effectiveness(train_raw, min_battles=15, alpha=1.5, beta=1.5,
                                    min_cap=0.75, max_cap=1.25)
    train_df = build_feature_df(train_raw, te_full)
    test_df  = build_feature_df(test_raw, te_full)

    # We drop columns with no variation and make sure both datasets have the same structure.
    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) <= 1]
    if const_cols:
        train_df = train_df.drop(columns=const_cols)
        test_df  = test_df.drop(columns=[c for c in const_cols if c in test_df.columns])
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0.0)

    features = [c for c in train_df.columns if c not in ['battle_id', 'player_won']]
    X = train_df[features].astype(float).values
    y = train_df['player_won'].astype(int).values
    X_test = test_df[features].astype(float).values

    # We train the final VotingClassifier using all the training data to get the best possible model
    voter_final = VotingClassifier(
        estimators=[("xgb", make_xgb()), ("ada", make_adaboost())],
        voting="soft",
        weights=[0.5, 0.5]
    )

    voter_final.fit(X, y)

    # We make predictions on the test set using the final model to see how it performs on new data.
    p = voter_final.predict_proba(X_test)[:, 1]
    pred_test = (p >= 0.5).astype(int)

    # Final submission
    submission = pd.DataFrame({
        'battle_id': test_df['battle_id'].astype(int),
        'player_won': pred_test
    })
    submission.to_csv('submission_voting_xgb_ada.csv', index=False)

if __name__ == "__main__":
    run_voting()
