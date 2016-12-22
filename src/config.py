class Model_Config:
    """model configuration"""
    # regressor type
    # lin: linear regression using MSE as loss function
    # msvr: multiple support vector regression
    # prob: probabilistic regression (use the quantile regression algorithms)
    regressor = "lin"

    # msvr params
    epsilon = 15
    C  = 0.5

    # quantile regression params
    quantile_rate = 0.3

    # network params
    # the hidden size for three modalities, second level lstm and fully connected newwork
    n_first_hidden = 2048
    n_second_hidden = 1024
    n_third_hidden = 1024
    cnn_feat_size = 512
    n_hidden_level2 = 2048
    n_fully_connect_hidden = 4096

    # learning rate
    lr = 0.0001

    # data reader configuration
    # data_step: the lag between the two input series
    # n_step: the length of the input time series
    # h_head: the lag between the input and output time series
    # n_target: the lenght of the output length if not equal one, use the msvr
    batch_size = 100
    data_step = 24
    n_step = 72
    h_ahead = 9
    n_target = 1

    #train and test params
    epoch_size = 3000
    test_num = 70
    print_step = 10
    test_step = 200

    width = 64
    height = 64
