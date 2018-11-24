"""
contains all methods to import data in this project
"""
import yaml
import pandas as pd

class Importer:
    @staticmethod
    def get_cfg():
        """Loads configuration from yaml file"""
        return yaml.load((open('kddCup/config.yml', 'br')))

    @staticmethod
    def get_raw_dat(cfg):
        """Load raw data as a panda data frame"""
        return pd.read_csv('kddCup/data/' + cfg['data_file'], sep=',',
                           error_bad_lines=False, low_memory=False,
                           keep_default_na=True, verbose=True)
    @staticmethod
    def get_test_dat(cfg):
        """Load raw data as a panda data frame"""
        return pd.read_csv('kddCup/data/' + cfg['test_file'], sep=',',
                           error_bad_lines=False, low_memory=False,
                           keep_default_na=True, verbose=True)