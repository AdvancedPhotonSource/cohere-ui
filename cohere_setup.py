import sys
import os
import argparse
from shutil import copy
import re
import time


def getspecfile_datetime(file):
    SPEC_datetime = re.compile(r"^#D")
    SPEC_time_format = re.compile(r"\d\d:\d\d:\d\d")
    SPEC_multi_blank = re.compile(r"\s+")
    datetimestruct = None

    datetimeformat = "%a %b %d %H:%M:%S %Y"
    with open(file) as fid:
        for line in fid:
            if SPEC_datetime.match(line):
                filetime = SPEC_time_format.findall(line)[0]
                line = SPEC_datetime.sub("", line)
                line = SPEC_multi_blank.sub(" ", line).strip()
                # print(line)
                datetimestruct = time.strptime(line, datetimeformat)
                break

    return datetimestruct


def setup(script_dir, working_dir, det_name, specfile):
    # spec_timestamp = getspecfile_datetime(specfile)
    # spec_date = str(spec_timestamp.tm_year) + str(spec_timestamp.tm_mon) + str(spec_timestamp.tm_mday)
    # print(spec_date)

    exp_data_dir, spec_filename = os.path.split(specfile)
    specfilebase = os.path.splitext(spec_filename)[0]
    addatadir = os.path.join(exp_data_dir, "AD" + det_name + "_" + specfilebase)

    templ_confdir = os.path.join(script_dir, 'cohere-defaults')
    # conf files
    templ_configfile = os.path.join(templ_confdir, "config")
    templ_configprepfile = os.path.join(templ_confdir, "config_prep")
    templ_configdatafile = os.path.join(templ_confdir, "config_data")
    templ_configdispfile = os.path.join(templ_confdir, "config_disp")
    templ_configrecfile = os.path.join(templ_confdir, "config_rec")

    confdir = os.path.join(working_dir, 'cohere-defaults', 'conf')
    if not os.path.exists(confdir):
        os.makedirs(confdir)
    # conf files
    configfile = os.path.join(confdir, "config")
    configprepfile = os.path.join(confdir, "config_prep")
    configdatafile = os.path.join(confdir, "config_data")
    configdispfile = os.path.join(confdir, "config_disp")
    configrecfile = os.path.join(confdir, "config_rec")

    detcorrectionsdir = os.path.join(script_dir, 'cohere-scripts', 'beamlines', 'aps_34idc', 'detector_corrections', det_name)
    # figure out the dark and white files
    if os.path.isdir(detcorrectionsdir):
        darkfieldfile = os.path.join(detcorrectionsdir, "current_dark.tif")
        whitefieldfile = os.path.join(detcorrectionsdir, "current_white.tif")
    else:
        darkfieldfile = ""
        whitefieldfile = ""

    with open(templ_configfile, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('ANLDIR', working_dir)
    filedata = filedata.replace('SPECFILE', specfile)
    with open(configfile, 'a') as file:
        file.write(filedata)

    with open(templ_configprepfile, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('ADDATADIR', addatadir)
    filedata = filedata.replace('DARKFIELD', darkfieldfile)
    filedata = filedata.replace('WHITEFIELD', whitefieldfile)
    with open(configprepfile, 'a') as file:
        file.write(filedata)

    with open(templ_configdispfile, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('ANLDIR', working_dir)
    with open(configdispfile, 'a') as file:
        file.write(filedata)

    copy(templ_configdatafile, configdatafile)
    copy(templ_configrecfile, configrecfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", help="working directory")
    parser.add_argument("det_name", help="detector name")
    parser.add_argument("specfile", help="spec file including absolute path")
    args = parser.parse_args()

    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    det_name = args.det_name
    if (det_name.endswith(':')):
        det_name = det_name[:-1]

    setup(script_dir, args.working_dir, det_name, args.specfile)
