import sys
import os
import argparse
from shutil import copy
import re
import time
import glob


def getspecfile_datetime(file):
    SPEC_datetime = re.compile(r"^#D")
    datestr = None

    datetimeformat = "%a %b %d %H:%M:%S %Y"
    with open(file) as fid:
        for line in fid:
            if SPEC_datetime.match(line):
                line = SPEC_datetime.sub("", line).strip()
                datetimestruct = time.strptime(line, datetimeformat)
                mon = datetimestruct.tm_mon
                if mon < 10:
                    mon = '0' + str(mon)
                else:
                    mon = str(mon)
                mday = datetimestruct.tm_mday
                if mday < 10:
                    mday = '0' + str(mday)
                else:
                    mday = str(mday)
                datestr = str(datetimestruct.tm_year) + mon + mday
                break

    return datestr


def get_dark(detcorrectionsdir, spec_timestamp):
    files = glob.glob1(detcorrectionsdir, '*darkfield.tif')
    files.sort()
    filestamps = [file[8:16] for file in files]
    filestamps.append(spec_timestamp)
    filestamps.sort()
    ind = filestamps.index(spec_timestamp)
    # check for case the stamp is the same
    if len(filestamps) > ind + 1 and filestamps[ind] == filestamps[ind + 1]:
        return (os.path.join(detcorrectionsdir, files[ind]))
    elif ind == 0:
        return ""
    return(os.path.join(detcorrectionsdir, files[ind-1]))


def get_white(detcorrectionsdir, spec_timestamp):
    files = glob.glob1(detcorrectionsdir, '*whitefield.tif')
    files.sort()
    filestamps = [file[8:16] for file in files]
    filestamps.append(spec_timestamp)
    filestamps.sort()
    ind = filestamps.index(spec_timestamp)
    # check for case the stamp is the same
    if len(filestamps) > ind + 1 and filestamps[ind] == filestamps[ind + 1]:
        return (os.path.join(detcorrectionsdir, files[ind]))
    elif ind == 0:
        return ""
    return(os.path.join(detcorrectionsdir, files[ind-1]))


def setup(script_dir, working_dir, det_name, specfile):
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

    confdir = os.path.join(script_dir, 'cohere-defaults', 'conf')
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
        spec_timestamp = getspecfile_datetime(specfile)
        darkfieldfile = get_dark(detcorrectionsdir, spec_timestamp)
        whitefieldfile = get_white(detcorrectionsdir, spec_timestamp)
    else:
        darkfieldfile = ""
        whitefieldfile = ""

    with open(templ_configfile, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('ANLDIR', working_dir)
    filedata = filedata.replace('SPECFILE', specfile)
    with open(configfile, 'w') as file:
        file.write(filedata)

    with open(templ_configprepfile, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('ADDATADIR', addatadir)
    filedata = filedata.replace('DARKFIELD', darkfieldfile)
    filedata = filedata.replace('WHITEFIELD', whitefieldfile)
    with open(configprepfile, 'w') as file:
        file.write(filedata)

    with open(templ_configdispfile, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('ANLDIR', working_dir)
    with open(configdispfile, 'w') as file:
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
