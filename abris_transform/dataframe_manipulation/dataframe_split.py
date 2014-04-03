from random import shuffle


def split_dataframe_train_test(data_frame, train_percentage=0.7):

    train, cv, test = split_dataframe(data_frame,
                                      train_percentage=train_percentage,
                                      cv_percentage=0,
                                      test_percentage=max(0, 1 - train_percentage))
    return train, test


def split_dataframe(df, train_percentage=0.6, cv_percentage=0.2, test_percentage=0.2):
    """
    @return training, cv, test
            (as pandas dataframes)

    @param df: pandas dataframe
    @param train_percentage: float | percentage of data for training set (default=0.6)
    @param cv_percentage:    float | percentage of data for cross validation set (default=0.2)
    @param test_percentage:  float | percentage of data for test set (default=0.2)
    """
    sum_percentages = sum((train_percentage, cv_percentage, test_percentage))
    assert abs(sum_percentages - 1) < 0.001, \
        "Sum of training, cv and test_percentage should be 1. Was %f" % sum_percentages

    N = len(df)
    l = range(N)
    shuffle(l)

    # get splitting indices
    train_len = int(round(N * train_percentage))
    cv_len = int(round(N * cv_percentage))
    test_len = int(round(N * test_percentage))

    # get training, cv, and test sets
    training = df.ix[l[:train_len]]
    cv = df.ix[l[train_len:train_len + cv_len]]
    test = df.ix[l[train_len + cv_len:]]

    #print len(cl), len(training), len(cv), len(test)

    return training, cv, test
