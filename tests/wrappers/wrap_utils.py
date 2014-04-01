from tests.wrappers.config_wrapper import ConfigWrapper


def adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=None):
    """
    Returns a configuration instance after adding the dm_wrapper data into it.
    If config_wrapper is given, first adds the dm_wrapper data into it, and then
    returns the config from it.
    """
    if config_wrapper is None:
        config_wrapper = ConfigWrapper()
    config_wrapper.add_data_model_wrapper(dm_wrapper)

    config = config_wrapper.build_to_config()
    config.get_data_model().set_features_types_from_dataframe(dm_wrapper.dataframe)

    return config
