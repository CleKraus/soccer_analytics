
import helper.test as test_help
import ruamel.yaml


def test_read_config():

    config = test_help._get_config_file()

    assert type(config) == ruamel.yaml.comments.CommentedMap

