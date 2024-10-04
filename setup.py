import os

cur_dir = os.getcwd().replace(os.sep, '/')

with open('example_workspace/scan_54/conf/config', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config', 'w') as file:
    file.write(filedata)

with open('example_workspace/scan_54/conf/config_prep', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config_prep', 'w') as file:
    file.write(filedata)

with open('example_workspace/scan_54/conf/config_disp', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config_disp', 'w') as file:
    file.write(filedata)

with open('example_workspace/scan_54/conf/config_instr', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config_instr', 'w') as file:
    file.write(filedata)

with open('example_workspace/esrf_exp_4/conf/config', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/esrf_exp_4/conf/config', 'w') as file:
    file.write(filedata)
