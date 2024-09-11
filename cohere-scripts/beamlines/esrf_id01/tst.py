import h5py
import numpy as np







dfile = '/home/beams/CXDUSER/34idc-data/2024/Li724/Data/HEA_5515_old/HEA_5515_old_BCDI_0002/HEA_5515_old_BCDI_0002.h5'
dsets = []
def visit_func(name, node) :
    if 'pixel' in name:
        print ('Object:', node.name)

    
with h5py.File(dfile, 'r') as h5r:     
    h5r['2.1'].visititems(visit_func)

    for ds in dsets:
        data = h5r.get(ds)
        data = np.array(data)
        print('Shape, type: \n', ds, data.shape, data.dtype, np.mean(data))



scan = '1-2'
scans = [[int(x) for x in u.split('-')] for u in scan.replace(' ','').split(',')]
scans = [1-3]

h5file = h5py.File(dfile)
# roi = config["roi"]
roi = (516, 516)
rvx = []
print('in tst')
#d = h5file["/2.1/instrument/mpxgaas/image"]
d = h5file["/2.1/measurement/mpxgaas"]
data = np.array(d)
print('d', type(data), data.shape, data.dtype)

# for each peak:
#for (h, k, l), (start, stop) in zip(hkls, scans):
#for (start, stop) in scans:
    #Path(f"{experiment_dir}/mp_{h}{k}{l}").mkdir(exist_ok=True)
    #Path(f"{experiment_dir}/mp_{h}{k}{l}/preprocessed_data").mkdir(exist_ok=True)
    # grab the full scan range that corresponds to this peak
images = [h5file[f"{i}.1/measurement/mpx1x4"] for i in range(scans[0], scans[-1]+1)]
images = h5file["1.1/measurement/mpx1x4"]
print(type(images))

# identify which scans are rocking curves rather than realignment
num_frames = max([len(im) for im in images])
# Sum all the rocking curves for this peak
# extract the ROI from the image stack
stack = np.sum([im[:, roi[2]:roi[3], roi[0]:roi[1]] for im in images if len(im)==num_frames], axis=0)

print(stack.shape, stack.dtype)
# Do some basic preprocessing (maybe just threshold?)
#stack = np.clip(stack-config['intensity_threshold'], 0, None)
# Save the preprocessed_data
ut.save_tif(stack, f"{experiment_dir}/mp_{h}{k}{l}/preprocessed_data/prep_data.tif")
# Calculate the resampling matrix
instr_obj = instr.Instrument()
instr_obj.initialize(ut.read_config(f"{experiment_dir}/conf/config_instr"), start)
B_recip, _ = instr_obj.get_geometry(stack.shape, xtal=True)
B_recip = np.stack([B_recip[1, :], B_recip[0, :], B_recip[2, :]])
rs_voxel_size = np.min([np.linalg.norm(B_recip[:, i]) for i in range(3)])  # Units are inverse nanometers
rvx.append(rs_voxel_size)
