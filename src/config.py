class Model_Config:
    """model configuration"""
    # regressor type
    # lin: linear regression using MSE as loss function
    # msvr: multiple support vector regression
    # quantile: probabilistic regression (use the quantile regression algorithms)
    regressor = "lin"

    # msvr params
    epsilon = 10
    C  = 5.0

    # quantile regression params
    quantile_rate = 0.3

    # network params
    # the hidden size for three modalities, second level lstm and fully connected newwork
    n_first_hidden = 100
    n_second_hidden = 500
    n_third_hidden = 100
    cnn_feat_size = 200
    n_hidden_level2 = 400

    # learning rate
    lr = 0.005

    #modal configuration
    modality = [0, 1, 0]

    # data reader configuration
    # data_step: the lag between the two input series
    # n_step: the length of the input time series
    # h_head: the lag between the input and output time series
    # n_target: the lenght of the output length if not equal one, use the msvr
    batch_size = 100
    data_step = 24

    n_step_1 = 72
    n_step_2 = 24
    n_step_3 = 72

    n_shift_1 = 72
    n_shift_2 = 0
    n_shift_3 = 72

    h_ahead = 9
    n_target = 1

    #train and test params
    epoch_size = 10000

    print_step = 100
    test_step = 200

    n_input_ir = 61
    n_input_mete = 30
    width = 64
    height = 64
