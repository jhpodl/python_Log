import configparser
import os

config = configparser.ConfigParser()
config['DEV'] = {'ip': '10.180.10.50', 'USERNAME': 'crdhkim', 'PASSWORD': 'C0mpassion23$', 'PORT': 22}
config['PROD1'] = {'ip': '10.180.10.61', 'USERNAME': 'prodwas1', 'PASSWORD': 'ZjaVoTus12#$', 'PORT': 22}
config['PROD2'] = {'ip': '10.180.10.62', 'USERNAME': 'prodwas2', 'PASSWORD': 'ZjaVoTus12#$', 'PORT': 22}

config['LOG_PATHS'] = {'ORIGIN_LOG_PATH': "/data/logs/catalina/",
                       'SOURCE_LOG_PATH': os.path.expanduser("~/Documents/"),
                       'result': os.path.expanduser("~/Documents/result.txt")}

config['MESSAGE'] = {'download': "Download selected file",
                     'search_text': "Please enter a search word above entry",
                     'combo_search': "Search an item..."}

config['COMMAND'] = {'ls_command': "ls /data/logs/catalina/"}

with open('config.ini', 'w') as configfile:
    config.write(configfile)
