import os
import cohere_core.utilities as ut


def process_batch(get_scan_array_function, scans_datainfo, experiment_dir, separate_scan_ranges):
    """

    :param get_scan_array_function: instr_obj.get_scan_array
    :param scans_datainfo: info that will allow to obtain the data array using the above function
    :param experiment_dir: directory to save the correlation error reporting in a case of combining
        scans. Not used in this simple example.
    :return:
    """
    print('scans_datainfo', scans_datainfo)
    print('get_scan_array_function', get_scan_array_function)
    arr = get_scan_array_function(scans_datainfo[1])
    # save the file
    save_dir = ut.join(experiment_dir, 'preprocessed_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(f"Saving array (max={int(arr.max())}) as {save_dir + '/' + filename}")
    save_file = ut.join(save_dir, 'prep_data.tif')
    ut.save_tif(arr, save_file)


