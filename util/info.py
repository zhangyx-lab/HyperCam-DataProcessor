from configparser import ConfigParser
from glob import glob as glob
# Project Packages
from env import BASE, DATA_PATH
# Path to load and store runtime info
RUNTIME_INFO_PATH = DATA_PATH / 'runtime.ini'
# Read all info.ini files in data folder (and subfolders)
INFO_INI = ConfigParser()


def load_info(*srcs):
    srcs = list(srcs)
    # Reload ini file list
    if len(srcs) == 0:
        # Default values, will be overwritten by subsequent files
        srcs += list(glob(str(BASE / 'templates' / '*.ini')))
        # Find info.ini in user data
        srcs += list(glob(str(DATA_PATH / '**' / 'info.ini')))
        # runtime generated information
        srcs += list(glob(str(RUNTIME_INFO_PATH)))
    INFO_INI.read(srcs)


def INFO(section, key=None, fmt=str, reload=False):
    def GET_INFO(key, fmt=fmt, optional=False):
        if reload:
            load_info()
        try:
            return fmt(INFO_INI.get(section, key))
        except:
            assert optional == True, f"Unable to find '{key}' of ini section [{section}]"
            return None
    # Return wrapped info
    if key is None:
        return GET_INFO
    else:
        return GET_INFO(key)


load_info()


def runtime_info_init():
    with open(BASE / 'templates' / 'runtime.ini', 'r') as template:
        t = template.readlines()
        with open(RUNTIME_INFO_PATH, 'w') as runtime:
            runtime.writelines(t)


def runtime_info(key, val):
    with open(RUNTIME_INFO_PATH, 'a') as runtime:
        runtime.write(f"{key} = {val}\n")
    # Reload info to make it take effect immediately
    load_info(RUNTIME_INFO_PATH)


def runtime_log(*contents):
    def comment(content):
        return "# " + str(content) + "\n"
    with open(RUNTIME_INFO_PATH, 'a') as runtime:
        runtime.writelines(map(comment, contents))
