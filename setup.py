import os

cur_dir = os.getcwd()

with open('workspace/t_54/conf/config', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('workspace/t_54/conf/config', 'w') as file:
    file.write(filedata)

with open('workspace/t_54/conf/config_prep', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('workspace/t_54/conf/config_prep', 'w') as file:
    file.write(filedata)

