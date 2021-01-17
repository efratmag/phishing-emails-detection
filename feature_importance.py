import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.inspection import permutation_importance

def get_feature_permutated_importance(model, excluded_final_protocol=[]):
    protocols = np.unique(X_train_by_protocol.Protocol)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    feature_importance = {}
    for i, (train, test) in enumerate(cv.split(X_train_pcap, y_train_pcap)):
        pcap_train = X_train_pcap[train]
        pcap_test = X_train_pcap[test]
        protocol_count = protocols.shape[0]
        for j in range(protocol_count):
            protocol = protocols[j]

            train_idx = np.where((X_train_by_protocol.Protocol==protocol) & (X_train_by_protocol.pcap_id.isin(pcap_train)))[0]
            test_idx = np.where((X_train_by_protocol.Protocol==protocol) & (X_train_by_protocol.pcap_id.isin(pcap_test)))[0]

            if train_idx.shape[0] == 0 or test_idx.shape[0] == 0:
                continue
            X_train = X_train_by_protocol.loc[train_idx].drop(columns=['pcap_id','Protocol'])
            y_train = y_train_by_protocol[train_idx]
            X_test = X_train_by_protocol.loc[test_idx].drop(columns=['pcap_id', 'Protocol'])
            y_test = y_train_by_protocol[test_idx]

            if protocol not in feature_importance:
                feature_importance[protocol] = {}
                feature_importance[protocol]['importances_mean'] = []
                feature_importance[protocol]['importances_std'] = []
                feature_importance[protocol]['importances'] = []

            model.fit(X_train, y_train)
            r = permutation_importance(model, X_test, y_test, n_repeats=75, random_state=0, n_jobs=-1)
            feature_importance[protocol]['importances_mean'].append(r.importance_mean)
            feature_importance[protocol]['importances_std'].append(r.importance_std)
            feature_importance[protocol]['importances'].append(r.importances)

    return feature_importance




