# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import sys
import tifffile as tf
from functools import reduce

MEM_FACTOR = 170
GA_FAST_MEM_FACTOR = 184

conf_file = sys.argv[1]
data_file = sys.argv[2]
no_recs = int(sys.argv[3])
is_ga =  int(sys.argv[4])

data_shape = tf.imread(data_file).shape
data_size = reduce((lambda x, y: x * y), data_shape) / 1000000.

if no_recs > 1:
    if is_ga:
        mem_req = GA_FAST_MEM_FACTOR * data_size + 430
        script = 'reconstruction_ga.py'
    else:
        mem_req = MEM_FACTOR * data_size
        script = 'reconstruction_multi.py'
else:
    mem_req = data_size
    script = 'reconstruction_single.py'

# each node has four GPU with 40.5 GB
recs_on_gpu = 4050. // mem_req
reqs_on_node = int(recs_on_gpu * 4)

no_nodes = int(no_recs // reqs_on_node) + 1
ranks_per_node = reqs_on_node

print(no_nodes)

# replace ranks per node, number of ranks, and script in mpi_job.sh script
with open('mpi_job.sh', 'r') as file :
    filedata = file.read()
    filedata = filedata.replace('NRANKS', str(no_recs))
    filedata = filedata.replace('RANKS_PER_NODE', str(ranks_per_node))
    filedata = filedata.replace('SCRIPT', script)
    filedata = filedata.replace('CONF', conf_file)
    filedata = filedata.replace('DATA_FILE', data_file)
with open('A_mpi_job.sh', 'w') as file :
   file.write(filedata)
