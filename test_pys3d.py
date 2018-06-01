from pys3d import PYS3D
from itertools import product

if __name__ == '__main__':
    data_name = 'ring'
    #num_rows = 7400
    ''' test init '''
    s3d = PYS3D(data_name)

    ''' test fit '''
    #train_data_path = s3d.data_path + '0/train.csv'
    #print(train_data_path)
    #train_model_path = s3d.model_path + '0/'
    #print(train_model_path)
    #s3d.fit(train_data_path, train_model_path)

    ''' test predict '''
    #test_data_path = s3d.data_path + '0/test.csv'
    #print(test_data_path)
    #prediction_path = s3d.prediction_path + '0/'
    #s3d.predict(test_data_path, train_model_path, prediction_path)

    ''' test cross validation (inner) '''
    #fold_index = 1
    #lambda_ = 0.001
    #max_features = 3
    #print(s3d._inner_cross_validation_fold(fold_index, lambda_, max_features))

    ''' test cross validation '''
    #fold_index = 1
    #lambda_list = [0.0001, 0.003]
    #n_f_list = [3]
    #param_grid = list(product(lambda_list, n_f_list))
    #s3d._inner_cross_validation(1, param_grid)
    #print('done 1')
    #s3d._inner_cross_validation(2, param_grid)
    #print('done 2')

    ''' test cross validation across all folds '''
    lambda_list = [0.0001, 0.003]
    n_f_list = [3]
    param_grid = list(product(lambda_list, n_f_list))
    s3d.cross_val(param_grid)
