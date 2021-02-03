import pandas as pd
import os

from data import get_data
import feature_extraction
from constants import LABEL_COL, TEXT_COLS, config_file
from train_test_val_split import data_split
import configparser
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import warnings
from models import StackedModel, fit_catboost


from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier


warnings.filterwarnings('ignore')


def get_preprocessed_data(config_file, text_cols):
    """
    upload data and extract features
    """
    # preprocessing and feature extraction
    config = configparser.ConfigParser()
    config.read(config_file)
    data_dir = config["DEFAULT"]["data_dir"]
    data_files = config["DEFAULT"]["data_file_names"].split(",")
    train_X_raw, train_y_raw, test_X_raw = get_data(data_files, data_dir)

    common_words_subject_phishing, common_words_subject_ham = \
        feature_extraction.count_common_words_in_subject(train_X_raw, train_y_raw)
    X = feature_extraction.create_features(train_X_raw, text_cols,
                                           common_words_subject_phishing, common_words_subject_ham)
    y = train_y_raw[LABEL_COL]
    X_unlabeled = feature_extraction.create_features(test_X_raw, text_cols,
                                                     common_words_subject_phishing, common_words_subject_ham)

    return X, y, X_unlabeled


def make_prediction(X, y, test_X, model_packs, apply_chi2=True):

    X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y, TEXT_COLS)

    for fe_name, fe_params in feature_extractors.items():
        fe = fe_params['class'](**fe_params['args'])
        fe.fit(train_X)

        train_X_fe = fe.transform(train_X)
        validation_X_fe = fe.transform(validation_X)
        test_X_fe = fe.transform(test_X['description'])
        if isinstance(fe, TfidfVectorizer):
            train_X_fe = pd.DataFrame(train_X_fe.toarray(), columns=fe.get_feature_names(), index=train_X.index)
            validation_X_fe = pd.DataFrame(validation_X_fe.toarray(), columns=fe.get_feature_names(),
                                           index=validation_X.index)
            test_X_fe = pd.DataFrame(test_X_fe.toarray(), columns=fe.get_feature_names(), index=test_X.index)

        if apply_chi2:
            fs = SelectKBest(chi2, k=300)
            train_X_fe = fs.fit_transform(train_X_fe, train_y)
            validation_X_fe = fs.transform(validation_X_fe)
            test_X_fe = fs.transform(test_X_fe)

            train_X_fe = pd.DataFrame(train_X_fe, index=train_X.index)
            validation_X_fe = pd.DataFrame(validation_X_fe, index=validation_X.index)
            test_X_fe = pd.DataFrame(test_X_fe, index=test_X.index)

        stacked_model = StackedModel()
        for model_name, model_pack in model_packs.items():
            model = model_pack["class"](**model_pack["args"])
            if isinstance(model, CatBoostClassifier):
                print("eval catboost")
                res = model.grid_search(model_pack["hyper"], X=pd.concat([train_X_fe, validation_X_fe]),
                                        y=pd.concat([train_y, validation_y]))
                best_params = res['params']
                best_params['n_estimators'] = 5000
                model = model_pack["class"](**best_params)
                clf = fit_catboost(model, train_X_fe, train_y, validation_X_fe, validation_y)
                best_score = f1_score(model.predict(validation_X_fe), validation_y, average="weighted")
            else:
                clf = GridSearchCV(model, model_pack["hyper"], scoring='f1_weighted', cv=3)
                clf.fit(pd.concat([train_X_fe, validation_X_fe]), pd.concat([train_y, validation_y]))
                best_params = clf.best_params_
                best_score = clf.best_score_
                if model_name == 'AdaBoostClassifier':
                    best_params['n_estimators'] = 1000
                elif model_name == 'RandomForestClassifier':
                    best_params['n_estimators'] = 100
                clf.fit(train_X_fe, train_y)

            stacked_model.fit_estimator(train_X_fe, train_y, model_name, model_pack["class"], best_params,
                                        validation_X_fe, validation_y
                                        )
            preds = clf.predict(test_X_fe)
            test_X[LABEL_COL] = preds
            test_X.reset_index()[['index', LABEL_COL]].to_csv(
                os.path.join(RAW_DIR, f"{int(100 * best_score)}_{model_name}_{fe_name}_submission.csv"), index=False)
        stacked_model.fit_stacked(validation_X_fe, validation_y)
        preds = stacked_model.predict(test_X_fe)
        test_X[LABEL_COL] = preds
        test_X.reset_index()[['index', LABEL_COL]].to_csv(os.path.join(RAW_DIR, f"stacked_{fe_name}_submission.csv"),
                                                          index=False)


if __name__ == "__main__":
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    models = {''}
    log_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
    svc_model = make_pipeline(StandardScaler(), SVC())
    dtr_model = DecisionTreeClassifier()
    rfc_model = RandomForestClassifier()
    gnb_model = GaussianNB()
    xgb_model = XGBClassifier(eval_metric='error')
    adb_model = AdaBoostClassifier(n_estimators=100)

    # import data

    X, y, X_unlabeled = get_preprocessed_data(config_file, TEXT_COLS)


    output = make_prediction(train_df, X_unlabeled, model_packs_ensemble, feature_extractors, start_year=2017, end_year=2019,debug=True)
