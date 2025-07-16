# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import ast
import cohere_core.utilities as ut
import cohere_ui.beamlines.esrf_id01.beam_verifier as ver


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
        self.remove_outliers = QCheckBox('remove outliers')
        self.remove_outliers.setChecked(False)
        layout.addRow(self.remove_outliers)
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
        self.remove_outliers.setChecked('remove_outliers' in conf_map and conf_map['remove_outliers'])
        if 'outliers_scans' in conf_map:
            self.outliers_scans.setText(str(conf_map['outliers_scans']).replace(" ", ""))


    def clear_conf(self):
        self.exclude_scans.setText('')
        self.roi.setText('')
        self.remove_outliers.setChecked(False)
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
        if self.remove_outliers.isChecked():
            conf_map['remove_outliers'] = True
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

        if 'remove_outliers' in conf_map and conf_map['remove_outliers']:
            # exclude outliers_scans from saving
            current_prep_map = ut.read_config(ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))
            if current_prep_map is not None and 'outliers_scans' in current_prep_map:
                conf_map['outliers_scans'] = current_prep_map['outliers_scans']
        ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))

        self.tabs.run_prep()

        # reload the window if remove_outliers as the outliers_scans could change
        if 'remove_outliers' in conf_map and conf_map['remove_outliers']:
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

