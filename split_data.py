import os, sys, time
import pandas as pd

def split_data(data_path, data_name, num_folds,
               outfolder=None):
    ''' generate equal folds of data. this is mainly for s3d

        parameters
        ----------
        data_path : str
            input data
        data_name : str
            name of data; for saving
        num_folds : int
            the number of folds to use for cross validation
        outfolder : str
            where to output the splitted datasets

        the function will export each fold into `outfolder`
        with names formatted as `data_name_i.csv` where `i` is the fold index
    '''

    if outfolder is None:
        outfolder = 'splitted_data/{}'.format(data_name)

    data = pd.read_csv(data_path)
    nrows, ncols = data.shape
    ## shuffle indices
    indices = pd.np.random.choice(pd.np.arange(nrows), size=nrows, replace=False)
    ## split
    indices_split = pd.np.array_split(indices, num_folds)
    print('splitting {} data ({} rows) into {} folds'.format(data_name, nrows, num_folds))
    print('each fold will have approximately {} rows'.format(indices_split[0].size))
    ## export different folds
    start = time.time()
    for i, indices_subarr in enumerate(indices_split):
        print('working on fold {}'.format(i), end=' ')
        ## create the corresponding fold-folder to save test and train/tune datasets
        out = '{}/{}/'.format(outfolder, i)
        if not os.path.exists(out):
            os.makedirs(out)
        ## export test dataset
        data.iloc[indices_subarr].to_csv(out+'test.csv', index=False)
        ## export train/tune dataset
        train_indices = pd.np.setdiff1d(indices, indices_subarr, True)
        data.iloc[train_indices].to_csv(out+'train.csv', index=False)

        ## also export the number of rows for train/test into a text file
        with open(out+'num_rows.csv', 'w') as f:
            ## first train, then test
            f.write(str(train_indices.size)+'\n')
            f.write(str(indices_subarr.size)+'\n')
        assert train_indices.size+indices_subarr.size == nrows
        print('(elapsed time: {0:.2f} seconds)'.format(time.time()-start))

if __name__ == '__main__':
    '''
    ## test on split_data func
    data_name = 'ring'
    num_folds = 5
    split_data(data_path, data_name, num_folds,
               outfolder='splitted_data/')
   '''

    data_name, num_folds = sys.argv[1:3]
    data_path = './data/{}.csv'.format(data_name)

    split_data(data_path, data_name, int(num_folds))
