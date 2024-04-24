import os

cur_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
with open('mpi_job.sh', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('ROOT_DIR', root_dir)
with open('mpi_job.sh', 'w') as file:
    file.write(filedata)

