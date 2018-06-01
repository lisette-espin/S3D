import pandas as pd
from contextlib import redirect_stdout
import subprocess, os, shutil, time, utils, io

class PYS3D(object):
    ''' a wrapper function to run s3d in python
        make it similar to sklearn interfaces
    '''
    def __init__(self,
                 data_name,
                 data_path = 'splitted_data/',
                 model_path = 'models/',
                 prediction_path = 'predictions/'):
        ''' initializer

            parameters
            ----------
            data_path : str
                input data base path
            model_path : str
                base path for output of built models
            prediction_path : str
                base path for predicted expectations

            for each path, we assume that there are sub folders for each test fold
        '''

        print('...s3d initializing...')

        #self.num_rows = num_rows
        self.data_name = data_name
        self.data_path = data_path + self.data_name + '/'
        self.model_path = model_path + self.data_name + '/'
        self.prediction_path = prediction_path + self.data_name + '/'
        ## check path validity
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)

        ## find the number of folds by counting the number of folders in self.data_path
        self.num_folds = len(os.listdir(self.data_path))
        self.inner_num_folds = self.num_folds - 1

        ## create a temporary folder for each fold
        self.tmp_path = './tmp/{}/'.format(self.data_name)
        if os.path.exists(self.tmp_path):
            #os.rmdir(self.tmp_path)
            shutil.rmtree(self.tmp_path)
        os.mkdir(self.tmp_path)
        ## create a folder with similar structure for hyperparameter searching
        self.cv_path = './cv/{}/'.format(self.data_name)
        if os.path.exists(self.cv_path):
            #os.rmdir(self.cv_path)
            shutil.rmtree(self.cv_path)
        os.mkdir(self.cv_path)

        for fold_index in range(self.num_folds):
            tmp_path = self.tmp_path + str(fold_index) + '/'
            ## remove it, if exist
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            ## create it
            os.mkdir(tmp_path)
            ## create a cv folder for each fold
            cv_path = self.cv_path + str(fold_index) + '/'
            ## remove it, if exist
            if os.path.exists(cv_path):
                shutil.rmtree(cv_path)
            ## create it
            os.mkdir(cv_path)

        #print('s3d with {} data, with {} rows, splitted into {} folds'.format(data_name,
        #                                                                      self.num_rows,
        print('s3d with {} data, splitted into {} folds'.format(self.data_name, self.num_folds))
        print('data saved in {}'.format(self.data_path))
        print('built models are saved to {}'.format(self.model_path))
        print('predictions are saved to {}'.format(self.prediction_path))
        print('temporary folder in ', './tmp/{}'.format(self.data_name))
        print('...done initializing...\n')

    def fit(self, train_data_path, train_model_path,
            lambda_=0.01, max_features=None,
            start_skip_rows=-1, end_skip_rows=-1):
        ''' fit s3d with the given lambda value

            parameters
            ----------
            train_data_path : str
                training data file
            train_model_path : str
                training data file
            lambda_ : float
                regularization parameter
            max_features : int
                maximum number of features to choose (default 20)
        '''

        if not os.path.exists(train_model_path):
            os.makedirs(train_model_path)

        c = './train -infile:{0} -outfolder:{1} -lambda:{2} -ycol:0'.format(train_data_path,
                                                                            train_model_path,
                                                                            lambda_)
        c += ' -start_skip_rows:{} -end_skip_rows:{}'.format(start_skip_rows, end_skip_rows)
        if max_features is not None:
            c += ' -max_features:{}'.format(max_features)

        ## catch the output and save to a log file in the `outfolder`
        process = subprocess.Popen(c.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        with open(train_model_path+'fit.log', 'w') as logfile:
            logfile.write(c)
            logfile.write(output.decode('utf8'))
            logfile.write('---errors below---\n')
            if error is not None:
                logfile.write(error)

    def predict(self, test_data_path,
                train_model_path,
                prediction_path,
                max_features=None, min_samples=1,
                start_use_rows=0, end_use_rows=-1):
        ''' predict for the held-out set

            parameters
            ----------
            test_data_path : str
                test data file, for prediction
            train_model_path : str
                pre-trained model for prediction
            prediction_path : str
                prediction output path
            max_features : int
                maximum number of features used for prediction (default use all s3d chosen features))
            min_samples : int
                minimum number of samples required to make a prediction (default 1)
        '''

        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        c = './predict_expectations -datafile:{} -infolder:{} -outfolder:{}'.format(test_data_path,
                                                                                    train_model_path,
                                                                                    prediction_path
                                                                                   )
        c += ' -start_use_rows:{} -end_use_rows:{}'.format(start_use_rows, end_use_rows)

        if min_samples > 1:
            c += ' -min_samples:{}'.format(min_samples)
        if max_features is not None:
            c += ' -max_features:{}'.format(max_features)

        process = subprocess.Popen(c.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        with open(prediction_path+'predict.log', 'w') as logfile:
            logfile.write(c)
            logfile.write(output.decode('utf8'))
            logfile.write('---errors below---\n')
            if error is not None:
                logfile.write(error)

    def _inner_cross_validation_fold(self, fold_index,
                                     lambda_, max_features, calc_threshold=True):
        '''
            do a k-fold inner cross validation for a given outer fold: 1 fold for validation others train
            all data files are prepared beforehand by using `split_data` function
            this is an inner cv version for finding the best hyperparameters

            parameters
            ----------
            lambda_ : float
                hyperparam 1: lambda_ parameter
            max_features : int
                hyperparam 2: max numbers of features
            calc_threshold : bool
                whether or not to calculate threshold based on the training data. Default: True
        '''

        print('--- inner cv for ', fold_index, 'th outer fold using lambda={0:.5f} and n_f={1} ---'.format(lambda_, max_features))
        start = time.time()

        ## use a dataframe to save the performance
        result_df = list()

        ## save the temporary model and predictions into the temporary files
        subfolder = self.tmp_path+'{}/'.format(fold_index)
        #print(subfolder)
        ## train data of the `fold_index`-th fold
        train_data_path = self.data_path + '{}/train.csv'.format(fold_index)
        num_train_rows, _ = open(self.data_path + '{}/num_rows.csv'.format(fold_index)).readlines()
        num_train_rows = int(num_train_rows)
        #print(num_train_rows)
        ## scan through each fold
        for fold_i in range(self.inner_num_folds):
            ## for each fold, use start_skip_rows to split
            start_skip_rows = int((fold_i*num_train_rows) / self.inner_num_folds)
            end_skip_rows = int(((fold_i+1)*num_train_rows) / self.inner_num_folds)
            print(fold_i, '-th fold: start -', start_skip_rows, 'end -', end_skip_rows)
            ## train
            self.fit(train_data_path, subfolder,
                     lambda_, max_features,
                     start_skip_rows=start_skip_rows,
                     end_skip_rows=end_skip_rows)
            ## validate
            self.predict(train_data_path,
                         subfolder, subfolder, ## save model and export predictions to the temporary place
                         max_features=max_features,
                         start_use_rows=start_skip_rows,
                         end_use_rows=end_skip_rows)
            if calc_threshold:
                thres = self.calculate_disc_threshold(subfolder, max_features)
            else:
                thres = 0.5
            #print('flexible threhold:', thres)

            ## prediction score
            ## read y_true; note that we need to skip the first 0 to start_skip_rows-1
            ## and read end_skip_rows-start_skip_rows
            y_true = pd.read_csv(train_data_path, usecols=[0], squeeze=True,
                                 skiprows=pd.np.arange(start_skip_rows),
                                 nrows=end_skip_rows-start_skip_rows
                                ).values
            #print('y true size:', y_true.size)
            ## read y_pred
            y_pred = pd.np.loadtxt(subfolder+'predicted_expectations_MF_{}.csv'.format(max_features))
            y_pred = (y_pred >= thres).astype(int)
            #print('y pred size:', y_pred.size)
            #roc_auc_score, f1_score, accuracy_score
            #print(d)
            result_df.append(utils.obtain_metric(y_true, y_pred))
            #break
        ## convert to dataframe
        #result_df['lambda_'] = lambda_
        #result_df['max_features'] = max_features
        avg_performance = pd.DataFrame(result_df).mean()
        avg_performance['lambda_'] = lambda_
        avg_performance['max_features'] = max_features
        ## include parameters lambda_ and max_features

        print('--- done inner cv for ', fold_index, 'th outer fold elapsed time: {0:.2f} seconds ---'.format(time.time()-start))

        return avg_performance

    def _inner_cross_validation(self, fold_index, param_grid,
                                eval_metric='auc_micro',
                                calc_threshold=True):
        '''
            same as _inner_cross_validation_fold, but a grid of parameters are given (param_grid)
            does cross validation on all of them and finally use a given evaluation metric to compare
            pick the best parameter and test on the test fold
            the final output is the selected parameters, fold index, and various performance


            parameters
            ----------
            param_grid : list of list
                a list of parameter combinations
            eval_metric : str
                the performance metric used for evaluating prediction performance
            calc_threshold : bool
                whether or not to calculate threshold based on the training data. Default: True
        '''

        stringio = io.StringIO()
        with redirect_stdout(stringio):
            ## capture everything and redirect into a file
            avg_performance_df = pd.DataFrame()
            print('--- start hyperparameter search for fold', fold_index, '---')
            start = time.time()
            for lambda_, n_f in param_grid:
                ## average performance for this fold
                avg_performance = self._inner_cross_validation_fold(fold_index,
                                                                    lambda_, n_f,
                                                                    calc_threshold=True)
                #print(avg_performance)
                avg_performance_df = avg_performance_df.append(avg_performance,
                                                               ignore_index=True)
            print('--- done hyperparam search (elapsed time {0:.2f} seconds)---'.format(time.time()-start))
            ## pick the best param for this test fold `fold_index`
            best_idx = avg_performance_df[eval_metric].idxmax()
            best_lambda_, best_max_features = avg_performance_df.loc[best_idx, ['lambda_', 'max_features']]
            best_max_features = int(best_max_features)

            ## train the model using the full training set with the best lambda and max_features
            train_data_path = self.data_path + '{}/train.csv'.format(fold_index)
            train_model_path = self.model_path +'{}/'.format(fold_index)
            print('select best params for fold', fold_index, "using {}: ".format(eval_metric))
            print('lambda_:', best_lambda_)
            print('number of features:', best_max_features)
            print('now train the model again with the selected parameters')
            self.fit(train_data_path, train_model_path,
                     best_lambda_, best_max_features)
            print('training done...\nstart testing...')
            test_data_path = self.data_path + '{}/test.csv'.format(fold_index)
            prediction_path = self.prediction_path + '{}/'.format(fold_index)
            #print('test_data_path', test_data_path)
            #print('prediction_path', prediction_path)
            self.predict(test_data_path,
                         train_model_path,
                         prediction_path, max_features=best_max_features)

            if calc_threshold:
                thres = self.calculate_disc_threshold(train_model_path, best_max_features)
            else:
                thres = 0.5

            ## evaluation
            y_true = pd.read_csv(test_data_path, usecols=[0], squeeze=True).values
            #print('y true size:', y_true.size)
            ## read y_pred
            y_pred = pd.np.loadtxt(prediction_path+'predicted_expectations_MF_{}.csv'.format(int(best_max_features)))
            y_pred = (y_pred >= thres).astype(int)
            print('--- test set performance ---')
            test_performance = utils.obtain_metric(y_true, y_pred)
            test_performance['fold_index'] = fold_index
            test_performance['lambda_'] = best_lambda_
            test_performance['n_f'] = best_max_features
            print(test_performance)
            print('--- cross val for fold {} ends ---'.format(fold_index))

        with open(self.cv_path+'{}/cv.log'.format(fold_index), 'w') as ofile:
            ofile.write(stringio.getvalue())

        test_performance.to_csv(self.cv_path+'{}/performance.csv'.format(fold_index))

        return test_performance


    def cross_val(self, param_grid,
                  eval_metric='auc_micro',
                  calc_threshold=True):
        ''' grid search for hyperparameters for each fold
        '''

        test_performance_df = pd.DataFrame()
        print('--- cross validation on', self.data_name, 'data ---')
        start = time.time()
        for fold_index in range(self.num_folds):
            print('starting on fold', fold_index)
            fold_start = time.time()
            test_performance = self._inner_cross_validation(fold_index, param_grid,
                                                            eval_metric, calc_threshold)
            print('finish after {0:.2f} seconds'.format(time.time()-fold_start))
            test_performance_df = test_performance_df.append(test_performance, ignore_index=True)
        print('--- done cv; total elapsed time {0:.2f} seconds'.format(time.time()-start))
        print(test_performance_df.mean())
        test_performance_df.to_csv(self.cv_path+'test_performance.csv', index=False)

    def calculate_disc_threshold(self, subfolder, max_features):
        ''' this function from peter's code on dropbox
            pick a threshold such as the number of predicted one's in the training set
            will be no less than the actual number of ones in the training set
        '''

        # (i) read in data
        # read in the number of datapoints in each group
        with open(subfolder+'N_tree.csv', 'r') as f:
            for row, line in enumerate(f):
                if row == max_features:
                    ## number of datapoints in each group
                    num_s = pd.np.array([int(x) for x in line.split()[0].split(',')])
                    break
        # total number of datapoints
        num_tot = num_s.sum()
        #print('num_tot:', num_tot)
        # read in the average value of y per group
        with open(subfolder+'ybar_tree.csv', 'r') as f:
            for row, line in enumerate(f):
                if row == 0:
                    ## global y-bar
                    ybar = float(line.split()[0])
                if row == max_features:
                    ## average y (y-bar's) in each group
                    ys = pd.np.array([float(x) for x in line.split()[0].split(',')])
                    break
        # total number of 1's
        num_ones = num_tot*ybar
        #print('num_ones:', num_ones)

        # (ii)  sort the ybars and N's
        #sort_indices_1 = pd.np.flip(pd.np.argsort(ys), axis=0)
        sort_indices = pd.np.argsort(-ys)
        ybar_sorted = ys[sort_indices]

        # (iii) get the cumsum of the ns
        ## sort the number of datapoints by their y bar values (large -> small)
        ## then get the cumulative number of datapoints
        num_sorted_cum = pd.np.cumsum(num_s[sort_indices])

        # (iv) find the first cum num value which is greater than num_ones
        num_ones_thres = pd.np.argmax(num_sorted_cum >= num_ones)

        # (v) pick the next element as the threshold, now all elememts greater than that threshold will be chosen
        disc_thresh = ybar_sorted[num_ones_thres]

        return disc_thresh

