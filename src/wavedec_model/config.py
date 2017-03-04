class Model_Config:
    """model configuration"""
    # regressor type
    # mse: mean square error
    # meef: information theoretic learning
    # msvr: multiple support vector regression
    # quantile: probabilistic regression (use the quantile regression algorithms)
    regressor = "mse"

    # msvr params
    epsilon = 10
    C  = 3

    # quantile regression params
    quantile_rate = 0.3

    # wavelet decomposition params
    level = 3
    wavelet = 'db4'

    # network params
    # the hidden size for three modalities, second level lstm and fully connected network
    n_first_hidden = 100
    n_second_hidden = 200
    cnn_feat_size = 200
    n_hidden_level2 = 400

    # learning rate
    lr = 0.0005

    #modal configuration
    modality = [1, 0]

    # data reader configuration
    # data_step: the lag between the two input series
    # n_step: the length of the input time series
    # h_head: the lag between the input and output time series
    # n_target: the lenght of the output length if not equal one, use the msvr
    batch_size = 500
    data_step = 1

    n_step_1 = 72
    n_step_2 = 72

    n_shift_1 = 72
    n_shift_2 = 72

    h_ahead = 0
    n_target = 1

    #train and test params
    epoch_size = 10000

    print_step = 100
    test_step = 500

    n_input_ir = 42
    n_input_mete = 22
