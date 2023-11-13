

if configs.pretrained_path is not None:
    assert os.path.isfile(configs.pretrained_path), "=> no checkpoint found at '{}'".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=torch.device('cpu')))
    if logger is not None:
        logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))
