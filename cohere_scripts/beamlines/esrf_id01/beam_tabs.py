import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import ast
import cohere_core.utilities as ut
import beamlines.esrf_id01.beam_verifier as ver


def msg_window(text):
    """
    Shows message with requested information (text)).
    Parameters
    ----------
    text : str
        string that will show on the screen
    Returns
    -------
    noting
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setWindowTitle("Info")
    msg.exec_()


def select_file(start_dir):
    """
    Shows dialog interface allowing user to select file from file system.
    Parameters
    ----------
    start_dir : str
        directory where to start selecting the file
    Returns
    -------
    str
        name of selected file or None
    """
    start_dir = start_dir.replace(os.sep, '/')
    dialog = QFileDialog(None, 'select dir', start_dir)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setSidebarUrls([QUrl.fromLocalFile(start_dir)])
    if dialog.exec_() == QDialog.Accepted:
        return str(dialog.selectedFiles()[0]).replace(os.sep, '/')
    else:
        return None


def select_dir(start_dir):
    """
    Shows dialog interface allowing user to select directory from file system.
    Parameters
    ----------
    start_dir : str
        directory where to start selecting
    Returns
    -------
    str
        name of selected directory or None
    """
    start_dir = start_dir.replace(os.sep, '/')
    dialog = QFileDialog(None, 'select dir', start_dir)
    dialog.setFileMode(QFileDialog.DirectoryOnly)
    dialog.setSidebarUrls([QUrl.fromLocalFile(start_dir)])
    if dialog.exec_() == QDialog.Accepted:
        return str(dialog.selectedFiles()[0]).replace(os.sep, '/')
    else:
        return None


def set_overriden(item):
    """
    Helper function that will set the text color to black.
    Parameters
    ----------
    item : widget
    Returns
    -------
    nothing
    """
    item.setStyleSheet('color: black')


class PrepTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(PrepTab, self).__init__(parent)
        self.name = 'Prep Data'
        self.conf_name = 'config_prep'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'prep' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.tabs = tabs
        self.main_win = main_window
        layout = QFormLayout()
        self.exclude_scans = QLineEdit()
        layout.addRow("exclude scans", self.exclude_scans)
        self.roi = QLineEdit()
        layout.addRow("detector area (roi)", self.roi)
        self.outliers_scans = QLineEdit()
        layout.addRow("outliers scans", self.outliers_scans)

        cmd_layout = QHBoxLayout()
        self.set_prep_conf_from_button = QPushButton("Load prep conf from")
        self.set_prep_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.prep_button = QPushButton('prepare', self)
        self.prep_button.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_prep_conf_from_button)
        cmd_layout.addWidget(self.prep_button)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.prep_button.clicked.connect(self.run_tab)
        self.set_prep_conf_from_button.clicked.connect(self.load_prep_conf)


    def load_tab(self, conf_map):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : dict
            configuration (config_prep)
        Returns
        -------
        nothing
        """
        if 'exclude_scans' in conf_map:
            self.exclude_scans.setText(str(conf_map['exclude_scans']).replace(" ", ""))
        if 'roi' in conf_map:
            self.roi.setText(str(conf_map['roi']).replace(" ", ""))
            self.roi.setStyleSheet('color: black')
        if 'outliers_scans' in conf_map:
            self.outliers_scans.setText(str(conf_map['outliers_scans']).replace(" ", ""))


    def clear_conf(self):
        self.exclude_scans.setText('')
        self.roi.setText('')
        self.outliers_scans.setText('')


    def load_prep_conf(self):
        """
        TODO: combine all load conf files in one function
        It display a select dialog for user to select a configuration file for preparation. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        prep_file = select_file(os.getcwd())
        if prep_file is not None:
            conf_map = ut.read_config(prep_file.replace(os.sep, '/'))
            self.load_tab(conf_map)
        else:
            msg_window('info: no prep config file')


    def get_prep_config(self):
        """
        It reads parameters related to preparation from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}
        if len(self.exclude_scans.text()) > 0:
            conf_map['exclude_scans'] = ast.literal_eval(str(self.exclude_scans.text()).replace(os.linesep,''))
        if len(self.roi.text()) > 0:
            conf_map['roi'] = ast.literal_eval(str(self.roi.text()).replace(os.linesep,''))

        return conf_map


    def run_tab(self):
        """
        Reads the parameters needed by prep script. Saves the config_prep configuration file with parameters from
        the window and runs the prep script.

        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
            return
        elif not self.main_win.is_exp_set():
            msg_window('the experiment has changed, press "set experiment" button')
            return
        else:
            conf_map = self.get_prep_config()
        # verify that prep configuration is ok
        er_msg = ver.verify('config_prep', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
              return

        # all parameters are optional, file can be empty
        if len(conf_map) > 0:
            ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))

        main_config_map = ut.read_config(ut.join(self.main_win.experiment_dir, 'conf', 'config'))
        auto_data = 'auto_data' in main_config_map and main_config_map['auto_data']

        if auto_data:
            # exclude outliers_scans from saving
            current_prep_map = ut.read_config(ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))
            if current_prep_map is not None and 'outliers_scans' in current_prep_map:
                conf_map['outliers_scans'] = current_prep_map['outliers_scans']
        ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))

        msg = self.tabs.run_prep()
        if len(msg) > 0:
            msg_window(msg)
            return

        # reload the window if auto_data as the outliers_scans could change
        if auto_data:
            prep_map = ut.read_config(ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))
            self.load_tab(prep_map)


    def save_conf(self):
        if not self.main_win.is_exp_exists():
            msg_window('the experiment does not exist, cannot save the config_prep file')
            return

        conf_map = self.get_prep_config()
        # verify that prep configuration is ok
        er_msg = ver.verify('config_prep', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
              return
        if len(conf_map) > 0:
            ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))


    def notify(self):
        self.tabs.notify(**{})


class DispTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(DispTab, self).__init__(parent)
        self.name = 'Display'
        self.conf_name = 'config_disp'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'disp' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.tabs = tabs
        self.main_win = main_window

        self.results_dir = None

        layout = QFormLayout()
        self.result_dir_button = QPushButton()
        layout.addRow("phasing results directory", self.result_dir_button)
        self.make_twin = QCheckBox('make twin')
        self.make_twin.setChecked(False)
        layout.addWidget(self.make_twin)
        self.unwrap = QCheckBox('include unwrapped phase')
        self.unwrap.setChecked(False)
        layout.addWidget(self.unwrap)
        self.crop = QLineEdit()
        layout.addRow("crop", self.crop)
        self.rampups = QLineEdit()
        layout.addRow("ramp upscale", self.rampups)
        cmd_layout = QHBoxLayout()
        self.set_disp_conf_from_button = QPushButton("Load disp conf from")
        self.set_disp_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.config_disp = QPushButton('process display', self)
        self.config_disp.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_disp_conf_from_button)
        cmd_layout.addWidget(self.config_disp)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.result_dir_button.clicked.connect(self.set_res_dir)
        self.config_disp.clicked.connect(self.run_tab)
        self.set_disp_conf_from_button.clicked.connect(self.load_disp_conf)


    def load_tab(self, conf_map):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : dict
            configuration (config_disp)
        Returns
        -------
        nothing
        """
        # Do not update results dir, as it may point to a wrong experiment if
        # it's loaded from another

        if 'make_twin' in conf_map:
            make_twin = conf_map['make_twin']
            if make_twin:
                self.make_twin.setChecked(True)
            else:
                self.make_twin.setChecked(False)
        else:
            self.make_twin.setChecked(False)

        if 'unwrap' in conf_map:
            unwrap = conf_map['unwrap']
            if unwrap:
                self.unwrap.setChecked(True)
            else:
                self.unwrap.setChecked(False)
        else:
            self.unwrap.setChecked(False)

        if 'crop' in conf_map:
            self.crop.setText(str(conf_map['crop']).replace(" ", ""))
        if 'rampups' in conf_map:
            self.rampups.setText(str(conf_map['rampups']).replace(" ", ""))


    def clear_conf(self):
        self.result_dir_button.setText('')
        self.make_twin.setChecked(False)
        self.unwrap.setChecked(False)
        self.crop.setText('')
        self.rampups.setText('')


    def load_disp_conf(self):
        """
        It display a select dialog for user to select a configuration file. When selected, the parameters
        from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        disp_file = select_file(os.getcwd())
        if disp_file is not None:
            conf_map = ut.read_config(disp_file.replace(os.sep, '/'))
            self.load_tab(conf_map)
        else:
            msg_window('info: no disp config file')


    def get_disp_config(self):
        """
        It reads parameters related to visualization from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}
        if len(self.result_dir_button.text()) > 0:
            conf_map['results_dir'] = str(self.result_dir_button.text()).replace(os.sep, '/')
        if self.make_twin.isChecked():
            conf_map['make_twin'] = True
        if self.unwrap.isChecked():
            conf_map['unwrap'] = True
        if len(self.crop.text()) > 0:
            conf_map['crop'] = ast.literal_eval(str(self.crop.text()).replace(os.linesep, ''))
        if len(self.rampups.text()) > 0:
            conf_map['rampups'] = ast.literal_eval(str(self.rampups.text()).replace(os.linesep, ''))

        return conf_map


    def run_tab(self):
        """
        Reads the parameters needed by format display script. Saves the config_disp configuration file with parameters from the window and runs the display script.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
            return
        if not self.main_win.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
            return
        if len(str(self.result_dir_button.text())) == 0:
            msg_window('the results directory is not set')
            return

        self.save_conf()
        self.tabs.run_viz()


    def save_conf(self):
        if not self.main_win.is_exp_exists():
            msg_window('the experiment does not exist, cannot save the config_disp file')
            return

        conf_map = self.get_disp_config()
        er_msg = ver.verify('config_disp', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
                return
        if len(conf_map) > 0:
            ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_disp'))


    def update_tab(self, **args):
        """
        Results directory is a parameter in display tab. It defines a directory tree that the display script will
        search for reconstructed image files and will process them for visualization. This function initializes it in
        typical situation to experiment directory. In case of active genetic algorithm it will be initialized to the
        generation directory with best results, and in case of alternate reconstruction configuration, it will be
        initialized to the last directory where the results were saved.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        # if separate scans, all scans will be processed
        if self.main_win.separate_scans.isChecked() or \
                self.main_win.separate_scan_ranges.isChecked():
            results_dir = self.main_win.experiment_dir
        else:
            if 'rec_id' in args:
                rec_id = args['rec_id']
                if len(rec_id) > 0:
                    results_dir = ut.join(self.main_win.experiment_dir, f'results_phasing_{rec_id}')
                else:
                    results_dir = ut.join(self.main_win.experiment_dir, 'results_phasing')

        self.result_dir_button.setText(results_dir)
        self.result_dir_button.setStyleSheet("Text-align:left")


    def set_res_dir(self):
        """
        Results directory is a parameter in display tab. It defines a directory tree that the display script will
        search for reconstructed image files and will process them for visualization. This function displays the
        dialog selection window for the user to select the results directory.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        results_dir = select_dir(os.getcwd())
        if results_dir is not None:
            self.result_dir_button.setStyleSheet("Text-align:left")
            self.result_dir_button.setText(results_dir.replace(os.sep, '/'))
        else:
            msg_window('please select valid results directory')


class InstrTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(InstrTab, self).__init__(parent)
        self.name = 'Instrument'
        self.conf_name = 'config_instr'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'Instrument' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.tabs = tabs
        self.main_win = main_window

        tab_layout = QVBoxLayout()
        gen_layout = QFormLayout()
        self.detector_button = QLineEdit()
        gen_layout.addRow("detector name", self.detector_button)
        self.diffractometer = QLineEdit()
        gen_layout.addRow("diffractometer", self.diffractometer)
        self.h5file_button = QPushButton()
        gen_layout.addRow("h5file file", self.h5file_button)
        tab_layout.addLayout(gen_layout)
        cmd_layout = QHBoxLayout()
        self.set_instr_conf_from_button = QPushButton("Load instr conf from")
        self.set_instr_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.save_instr_conf = QPushButton('save config', self)
        self.save_instr_conf.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_instr_conf_from_button)
        cmd_layout.addWidget(self.save_instr_conf)
        tab_layout.addLayout(cmd_layout)
        tab_layout.addStretch()
        self.setLayout(tab_layout)

        self.h5file_button.clicked.connect(self.set_h5file)
        self.save_instr_conf.clicked.connect(self.save_conf)
        self.set_instr_conf_from_button.clicked.connect(self.load_instr_conf)


    def run_tab(self):
        pass


    def load_tab(self, conf_map):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : dict
            configuration (config_disp)
        Returns
        -------
        nothing
        """
        if 'detector' in conf_map:
            self.detector_button.setStyleSheet("Text-align:left")
            self.detector_button.setText(conf_map['detector'])
        else:
            self.detector_button.setText('')
        if 'diffractometer' in conf_map:
            diff = str(conf_map['diffractometer']).replace(" ", "")
            self.diffractometer.setText(diff)
        if 'h5file' in conf_map:
            h5file = conf_map['h5file']
            if os.path.isfile(h5file):
                self.h5file_button.setStyleSheet("Text-align:left")
                self.h5file_button.setText(h5file)
            else:
                msg_window(f'The h5file file {h5file} in config file does not exist')


    def set_h5file(self):
        """
        Calls selection dialog. The selected h5 file is parsed.
        The h5file is saved in config.
        Parameters
        ----------
        none
        Returns
        -------
        noting
        """
        h5file = select_file(os.getcwd())
        if h5file is not None:
            self.h5file_button.setStyleSheet("Text-align:left")
            self.h5file_button.setText(h5file)
        else:
            self.h5file_button.setText('')

        self.save_conf()


    def clear_conf(self):
        self.detector_button.setText('')
        self.diffractometer.setText('')
        self.h5file_button.setText('')


    def load_instr_conf(self):
        """
        It display a select dialog for user to select a configuration file. When selected, the parameters
        from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        instr_file = select_file(os.getcwd())
        if instr_file is not None:
            conf_map = ut.read_config(instr_file.replace(os.sep, '/'))
            self.load_tab(conf_map)
        else:
            msg_window('please select valid instrument config file')


    def get_instr_config(self):
        """
        It reads parameters related to instrument from the window into a dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}
        if len(self.detector_button.text()) > 0:
            conf_map['detector'] = str(self.detector_button.text()).strip()
        if len(self.diffractometer.text()) > 0:
            conf_map['diffractometer'] = str(self.diffractometer.text())
        if len(self.h5file_button.text()) > 0:
            conf_map['h5file'] = str(self.h5file_button.text())

        return conf_map


    def save_conf(self):
        """
        Reads the parameters needed by format display script. Saves the config_disp configuration file with parameters from the window and runs the display script.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.main_win.is_exp_exists():
            msg_window('the experiment does not exist, cannot save the config_instr file')
            return

        conf_map = self.get_instr_config()
        if len(conf_map) == 0:
            return

        # verify here
        er_msg = ver.verify('config_instr', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
                return

        ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_instr'))

