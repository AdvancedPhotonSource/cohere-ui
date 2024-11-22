import os
import cohere_core.utilities as ut


def process_batch(get_scan_array_function, scans_datainfo, save_file, experiment_dir):
    """

    :param get_scan_array_function: instr_obj.get_scan_array
    :param scans_datainfo: info that will allow to obtain the data array using the above function
    :param save_file: file name where to save the data
    :param experiment_dir: directory to save the correlation error reporting in a case of combining
        scans. Not used in this simple example.
    :return:
    """
    if len(scans_datainfo) == 1:
        arr = get_scan_array_function(scans_datainfo[0][1])
    else:
        print("This example is writiten for a simple case of a single scan.")
        raise RuntimeError
    # save the file
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(f"Saving array (max={int(arr.max())}) as {save_dir + '/' + filename}")
    ut.save_tif(arr, save_file)


