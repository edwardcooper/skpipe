import os
import errno
import pandas as pd
from sklearn.model_selection import train_test_split

def train_dev_test_split(data,yname,sizes = [0.7,0.2,0.1], subdir = "./data", random_state = 7):

    # create ./data folder if it is not exist.
    create_data_folder(subdir)

    # do the X, y split of the data.
    if yname not in data.columns:
        raise NameError("The {0} column is not in the dataframe.".format(yname))

    X = data.drop(columns=yname)
    y = data[yname]

    # split the data into train/dev/test with a proportion of listed in sizes.
    if abs(sum(sizes) - 1.0) > 1e-9:
        raise ValueError("The sum of the sizes are not 1. You need to input a three item list that sums up to 1.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizes[-1], random_state=random_state,stratify = y)
    # calculate the test_size for the remaining split.
    relative_size = sizes[1]/(sizes[0]+sizes[1])

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=relative_size, random_state=random_state,stratify = y_train)


    # concat the X, y from train, dev, test
    train_data = pd.concat([X_train,y_train], axis = 1)
    dev_data = pd.concat([X_dev, y_dev], axis = 1)
    test_data = pd.concat([X_test, y_test], axis = 1)

    # write the train/dev/test data into subdir aka ./data directory.
    train_file_path = subdir + "/train_data.csv"
    train_data.to_csv(train_file_path, index = False)
    print("Written train data to disk")

    dev_file_path = subdir + "/dev_data.csv"
    dev_data.to_csv(dev_file_path, index = False)
    print("Written dev data to disk")

    test_file_path = subdir + "/test_data.csv"
    test_data.to_csv(test_file_path, index = False)
    print("Written test data to disk")



    return train_data, X_train, y_train



def create_data_folder(file_path):
    # using errno to solve the race condition problem
    # see http://deepix.github.io/2017/02/02/eexists.html for more.
    if not os.path.exists(file_path):
        try:
            os.makedirs(file_path, 0o700)
            print("Made a new directory")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif os.path.exists(file_path):
        print("The subdir {0} exists.".format(file_path))


def mkdir_ml():
    create_data_folder("./1_EDA")
    create_data_folder("./2_Data_clean")
    create_data_folder("./3_Baseline_models")
    create_data_folder("./4_Fine_tuning")
    create_data_folder("./5_Model_selection")
    create_data_folder('./6_Ensemble')
