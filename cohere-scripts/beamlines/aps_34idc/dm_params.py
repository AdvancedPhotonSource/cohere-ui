class DM_params():
    """
    This clas holds beamline 34idc parameters or derives such, needed for automated data flows.
    """

    @staticmethod
    def get_diffractometer(**kwargs):
        """
        Returns diffractometer name. For 34idc it is hard coded here.
        :param kwargs:
        :return:
        """
        return '34idc'

    @staticmethod
    def get_corrections(working_dir, **kwargs):
        """
        Returns tuple of darkfield and whitefield file names including path, applicable to the experiment.
        :param working_dir:
        :param kwargs:
        :return:
        """
        return (working_dir + '/cohere-scripts/beamlines/aps_34idc/detector_corrections/34icdTIM2/current_darkfield.tif',
        working_dir + '/cohere-scripts/beamlines/aps_34idc/detector_corrections/34icdTIM2/current_whitefield.tif')

    @staticmethod
    def get_data_dir_spec(working_dir, **kwargs):
        """
        Returns tuple of directory with raw data and a spec file applicable to the experiment.
        :param working_dir:
        :param kwargs:
        :return:
        """
        exp_data_dir = working_dir.replace('34idc-work', '34idc-data').split('/Analysis')[0]
        if exp_data_dir.endswith('/'):
            exp_data_dir = exp_data_dir[:-1]
        exp_name = exp_data_dir.split('/')[-1]

        return (exp_data_dir + '/AD34idcTIM2_' + exp_name + 'a', exp_data_dir + '/' + exp_name + 'a.spec')

    @staticmethod
    def get_dm_data_dir(experiment_dir, **kwargs):
        """
        Finds a directory on the monitored root for the given experiment where the preprocessed data will
        be saved.

        :param experiment_dir:
        :return: directory to save preprocessed data for this experiment
        """
        dirs = experiment_dir.split('/')
        # this is specific for 34idc
        # dirs[-3] contains beamline experiment name and dirs[-1] contains cohere-ui experiment name
        return '/home/beams/CXDUSER/34idc-work/hpc_data/' + dirs[-3] + '/' + dirs[-1]


    @staticmethod
    def get_dm_results_dir(experiment_dir, **kwargs):
        """
        Finds a directory on the monitored root for the given experiment where the reconstructed results
        are delivered by work flow.

        :param experiment_dir:
        :return: directory to save preprocessed data for this experiment
        """
        dirs = experiment_dir.split('/')
        # this is specific for 34idc
        # dirs[-3] contains beamline experiment name and dirs[-1] contains cohere-ui experiment name
        return '/home/beams/CXDUSER/34idc-work/hpc_results/' + dirs[-3] + '/' + dirs[-1]
