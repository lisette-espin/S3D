from split_data import split_data
if __name__ == '__main__':
    ## test on split_data func
    data_name = 'ring'
    data_path = 'data/{}.csv'.format(data_name)
    num_folds = 5
    split_data(data_path, data_name, num_folds)
