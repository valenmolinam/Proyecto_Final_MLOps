def add_features(df):
    df['diff_old_new_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['diff_old_new_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    
    df['amount_to_orig_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_balance'] = df['amount'] / (df['oldbalanceDest'] + 1)

    return df
