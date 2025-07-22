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
import cohere_ui.beamlines.aps_34idc.beam_verifier as ver


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
        self.data_dir_button = QPushButton()
        layout.addRow("data directory", self.data_dir_button)
        self.dark_file_button = QPushButton()
        layout.addRow("darkfield file", self.dark_file_button)
        self.white_file_button = QPushButton()
        layout.addRow("whitefield file", self.white_file_button)
        self.roi = QLineEdit()
        layout.addRow("detector area (roi)", self.roi)
        self.Imult = QLineEdit()
        layout.addRow("Imult", self.Imult)
        self.min_frames = QLineEdit()
        layout.addRow("min files in scan", self.min_frames)
        self.exclude_scans = QLineEdit()
        layout.addRow("exclude scans", self.exclude_scans)
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
        self.data_dir_button.clicked.connect(self.set_data_dir)
        self.dark_file_button.clicked.connect(self.set_dark_file)
        self.white_file_button.clicked.connect(self.set_white_file)
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
        if 'data_dir' in conf_map:
            if os.path.isdir(conf_map['data_dir']):
                self.data_dir_button.setStyleSheet("Text-align:left")
                self.data_dir_button.setText(conf_map['data_dir'])
            else:
                msg_window(f'The data_dir directory in config_prep file {conf_map["data_dir"]} does not exist')
        else:
            self.data_dir_button.setText('')
        if 'darkfield_filename' in conf_map:
            if os.path.isfile(conf_map['darkfield_filename']):
                self.dark_file_button.setStyleSheet("Text-align:left")
                self.dark_file_button.setText(conf_map['darkfield_filename'])
            else:
                msg_window(f'The darkfield file {conf_map["darkfield_filename"]} in config_prep file does not exist')
                self.dark_file_button.setText('')
        else:
            self.dark_file_button.setText('')
        if 'whitefield_filename' in conf_map:
            if os.path.isfile(conf_map['whitefield_filename']):
                self.white_file_button.setStyleSheet("Text-align:left")
                self.white_file_button.setText(conf_map['whitefield_filename'])
            else:
                self.white_file_button.setText('')
                msg_window(f'The whitefield file {conf_map["whitefield_filename"]} in config_prep file does not exist')
        else:
            self.white_file_button.setText('')
        if 'Imult' in conf_map:
            self.Imult.setText(str(conf_map['Imult']).replace(" ", ""))
        if 'min_frames' in conf_map:
            self.min_frames.setText(str(conf_map['min_frames']).replace(" ", ""))
        if 'exclude_scans' in conf_map:
            self.exclude_scans.setText(str(conf_map['exclude_scans']).replace(" ", ""))
        self.remove_outliers.setChecked('remove_outliers' in conf_map and conf_map['remove_outliers'])
        if 'outliers_scans' in conf_map:
            self.outliers_scans.setText(str(conf_map['outliers_scans']).replace(" ", ""))
        if 'roi' in conf_map:
            self.roi.setText(str(conf_map['roi']).replace(" ", ""))
            self.roi.setStyleSheet('color: black')


    def clear_conf(self):
        self.data_dir_button.setText('')
        self.dark_file_button.setText('')
        self.white_file_button.setText('')
        self.Imult.setText('')
        self.min_frames.setText('')
        self.exclude_scans.setText('')
        self.outliers_scans.setText('')
        self.remove_outliers.setChecked(False)
        self.roi.setText('')


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
            msg_window('select valid prep config file')


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
        if len(self.data_dir_button.text().strip()) > 0:
            conf_map['data_dir'] = str(self.data_dir_button.text()).strip()
        if len(self.dark_file_button.text().strip()) > 0:
            conf_map['darkfield_filename'] = str(self.dark_file_button.text().strip())
        if len(self.white_file_button.text().strip()) > 0:
            conf_map['whitefield_filename'] = str(self.white_file_button.text().strip())
        if len(self.Imult.text()) > 0:
            conf_map['Imult'] = ast.literal_eval(str(self.Imult.text()).replace(os.linesep,''))
        if len(self.min_frames.text()) > 0:
            min_frames = ast.literal_eval(str(self.min_frames.text()))
            conf_map['min_frames'] = min_frames
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

        try:
            self.tabs.run_prep()
        except ValueError as e:
            msg_window(str(e))
            return

        # reload the window if remove_outliers as the outliers_scans could change
        if 'remove_outliers' in conf_map and conf_map['remove_outliers']:
            prep_map = ut.read_config(ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))
            self.load_tab(prep_map)


    def set_dark_file(self):
        """
        It display a select dialog for user to select a darkfield file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        darkfield_filename = select_file(os.getcwd().replace(os.sep, '/'))
        if darkfield_filename is not None:
            darkfield_filename = darkfield_filename.replace(os.sep, '/')
            self.dark_file_button.setStyleSheet("Text-align:left")
            self.dark_file_button.setText(darkfield_filename)
        else:
            self.dark_file_button.setText('')


    def set_white_file(self):
        """
        It display a select dialog for user to select a whitefield file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        whitefield_filename = select_file(os.getcwd().replace(os.sep, '/'))
        if whitefield_filename is not None:
            whitefield_filename = whitefield_filename.replace(os.sep, '/')
            self.white_file_button.setStyleSheet("Text-align:left")
            self.white_file_button.setText(whitefield_filename)
        else:
            self.white_file_button.setText('')


    def set_data_dir(self):
        """
        It display a select dialog for user to select a directory with raw data file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        data_dir = select_dir(os.getcwd().replace(os.sep, '/'))
        if data_dir is not None:
            self.data_dir_button.setStyleSheet("Text-align:left")
            self.data_dir_button.setText(data_dir)
        else:
            self.data_dir_button.setText('')


    def save_conf(self):
        if not self.main_win.is_exp_exists():
            msg_window('the experiment does not exist, cannot save the config_prep file')
            return

        conf_map = self.get_prep_config()
        er_msg = ver.verify('config_prep', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
                return
        if len(conf_map) > 0:
            ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_prep'))


    def notify(self):
        self.tabs.notify(**{})


class SubInstrTab():
    def init(self, instr_tab, main_window):
        """
        Creates and initializes the 'Instrument' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.main_window = main_window
        self.instr_tab = instr_tab

        self.spec_widget = QWidget()
        spec_layout = QFormLayout()
        self.spec_widget.setLayout(spec_layout)
        self.energy = QLineEdit()
        spec_layout.addRow("energy", self.energy)
        self.delta = QLineEdit()
        spec_layout.addRow("delta (deg)", self.delta)
        self.gamma = QLineEdit()
        spec_layout.addRow("gamma (deg)", self.gamma)
        self.detdist = QLineEdit()
        spec_layout.addRow("detdist (mm)", self.detdist)
        self.th = QLineEdit()
        spec_layout.addRow("th (deg)", self.th)
        self.chi = QLineEdit()
        spec_layout.addRow("chi (deg)", self.chi)
        self.phi = QLineEdit()
        spec_layout.addRow("phi (deg)", self.phi)
        self.scanmot = QLineEdit()
        spec_layout.addRow("scan motor", self.scanmot)
        self.scanmot_del = QLineEdit()
        spec_layout.addRow("scan motor delta", self.scanmot_del)
        self.detector = QLineEdit()
        spec_layout.addRow("detector", self.detector)

        self.energy.textChanged.connect(lambda: set_overriden(self.energy))
        self.delta.textChanged.connect(lambda: set_overriden(self.delta))
        self.gamma.textChanged.connect(lambda: set_overriden(self.gamma))
        self.detdist.textChanged.connect(lambda: set_overriden(self.detdist))
        self.th.textChanged.connect(lambda: set_overriden(self.th))
        self.chi.textChanged.connect(lambda: set_overriden(self.chi))
        self.phi.textChanged.connect(lambda: set_overriden(self.phi))
        self.scanmot.textChanged.connect(lambda: set_overriden(self.scanmot))
        self.scanmot_del.textChanged.connect(lambda: set_overriden(self.scanmot_del))
        self.detector.textChanged.connect(lambda: set_overriden(self.detector))


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
        self.parse_spec()

        # if parameters are configured, override the readings from spec file
        if 'energy' in conf_map:
            self.energy.setText(str(conf_map['energy']).replace(" ", ""))
            self.energy.setStyleSheet('color: black')
        if 'delta' in conf_map:
            self.delta.setText(str(conf_map['delta']).replace(" ", ""))
            self.delta.setStyleSheet('color: black')
        if 'gamma' in conf_map:
            self.gamma.setText(str(conf_map['gamma']).replace(" ", ""))
            self.gamma.setStyleSheet('color: black')
        if 'detdist' in conf_map:
            self.detdist.setText(str(conf_map['detdist']).replace(" ", ""))
            self.detdist.setStyleSheet('color: black')
        if 'th' in conf_map:
            self.th.setText(str(conf_map['th']).replace(" ", ""))
            self.th.setStyleSheet('color: black')
        if 'chi' in conf_map:
            self.chi.setText(str(conf_map['chi']).replace(" ", ""))
            self.chi.setStyleSheet('color: black')
        if 'phi' in conf_map:
            self.phi.setText(str(conf_map['phi']).replace(" ", ""))
            self.phi.setStyleSheet('color: black')
        if 'scanmot' in conf_map:
            self.scanmot.setText(str(conf_map['scanmot']).replace(" ", ""))
            self.scanmot.setStyleSheet('color: black')
        if 'scanmot_del' in conf_map:
            self.scanmot_del.setText(str(conf_map['scanmot_del']).replace(" ", ""))
            self.scanmot_del.setStyleSheet('color: black')
        if 'detector' in conf_map:
            self.detector.setText(str(conf_map['detector']).replace(" ", ""))
            self.detector.setStyleSheet('color: black')


    def clear_conf(self):
        self.energy.setText('')
        self.delta.setText('')
        self.gamma.setText('')
        self.detdist.setText('')
        self.th.setText('')
        self.chi.setText('')
        self.phi.setText('')
        self.scanmot.setText('')
        self.scanmot_del.setText('')
        self.detector.setText('')


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
        if len(self.energy.text()) > 0:
            conf_map['energy'] = ast.literal_eval(str(self.energy.text()))
        if len(self.delta.text()) > 0:
            conf_map['delta'] = ast.literal_eval(str(self.delta.text()))
        if len(self.gamma.text()) > 0:
            conf_map['gamma'] = ast.literal_eval(str(self.gamma.text()))
        if len(self.detdist.text()) > 0:
            conf_map['detdist'] = ast.literal_eval(str(self.detdist.text()))
        if len(self.th.text()) > 0:
            conf_map['th'] = ast.literal_eval(str(self.th.text()))
        if len(self.chi.text()) > 0:
            conf_map['chi'] = ast.literal_eval(str(self.chi.text()))
        if len(self.phi.text()) > 0:
            conf_map['phi'] = ast.literal_eval(str(self.phi.text()))
        if len(self.scanmot.text()) > 0:
            conf_map['scanmot'] = str(self.scanmot.text())
        if len(self.scanmot_del.text()) > 0:
            conf_map['scanmot_del'] = ast.literal_eval(str(self.scanmot_del.text()))
        if len(self.detector.text()) > 0:
            conf_map['detector'] = str(self.detector.text())

        return conf_map


    def parse_spec(self):
        """
        Calls utility function to parse spec file. Displas the parsed parameters in the window with blue text.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.main_window.loaded and not self.main_window.is_exp_set():
            return
        scan = str(self.main_window.scan_widget.text())
        if len(scan) == 0:
            msg_window ('cannot parse spec, scan not defined')
            return

        diffractometer = self.instr_tab.diffractometer.text()
        if len(diffractometer) == 0:
            msg_window ('cannot parse spec, diffractometer not defined')
            return

        specfile = self.instr_tab.spec_file_button.text()
        if len(specfile) == 0:
            msg_window ('cannot parse spec, specfile not defined')
            return

        import cohere_ui.beamlines.aps_34idc.diffractometers as diff

        try:
            diff_obj = diff.create_diffractometer(diffractometer, {'specfile' : specfile})
        except Exception as e:
            msg_window (str(e))
            return

        last_scan = int(scan.split('-')[-1].split(',')[-1])
        spec_dict = diff_obj.parse_spec(last_scan)
        if spec_dict is None:
            return
        if 'energy' in spec_dict:
            self.energy.setText(str(spec_dict['energy']))
            self.energy.setStyleSheet('color: blue')
        if 'delta' in spec_dict:
            self.delta.setText(str(spec_dict['delta']))
            self.delta.setStyleSheet('color: blue')
        if 'gamma' in spec_dict:
            self.gamma.setText(str(spec_dict['gamma']))
            self.gamma.setStyleSheet('color: blue')
        if 'th' in spec_dict:
            self.th.setText(str(spec_dict['th']))
            self.th.setStyleSheet('color: blue')
        if 'chi' in spec_dict:
            self.chi.setText(str(spec_dict['chi']))
            self.chi.setStyleSheet('color: blue')
        if 'phi' in spec_dict:
            self.phi.setText(str(spec_dict['phi']))
            self.phi.setStyleSheet('color: blue')
        if 'detdist' in spec_dict:
            self.detdist.setText(str(spec_dict['detdist']))
            self.detdist.setStyleSheet('color: blue')
        if 'scanmot' in spec_dict:
            self.scanmot.setText(str(spec_dict['scanmot']))
            self.scanmot.setStyleSheet('color: blue')
        if 'scanmot_del' in spec_dict:
            self.scanmot_del.setText(str(spec_dict['scanmot_del']))
            self.scanmot_del.setStyleSheet('color: blue')
        if 'detector' in spec_dict:
            self.detector.setText(str(spec_dict['detector']))
            self.detector.setStyleSheet('color: blue')



class InstrTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(InstrTab, self).__init__(parent)
        self.name = 'Instrument'
        self.conf_name = 'config_instr'


    def toggle_config(self):
        if self.main_win.multipeak.isChecked() or self.main_win.separate_scans.isChecked() or self.main_win.separate_scan_ranges.isChecked():
            self.add_config = False
            self.extended.clear_conf()
            self.extended.spec_widget.hide()
        else:
            self.add_config = True
            self.extended.spec_widget.show()
            self.extended.parse_spec()
        if self.main_win.loaded:
            self.save_conf()


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
        self.extended = None
        if main_window.multipeak.isChecked() or main_window.separate_scans.isChecked() or main_window.separate_scan_ranges.isChecked():
            self.add_config = False
        else:
            self.add_config = True
        self.extended = SubInstrTab()
        self.extended.init(self, main_window)

        tab_layout = QVBoxLayout()
        gen_layout = QFormLayout()
        self.diffractometer = QLineEdit()
        gen_layout.addRow("diffractometer", self.diffractometer)
        self.spec_file_button = QPushButton()
        gen_layout.addRow("spec file", self.spec_file_button)
        tab_layout.addLayout(gen_layout)
        tab_layout.addWidget(self.extended.spec_widget)
        if not self.add_config:
            self.extended.spec_widget.hide()
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

        self.spec_file_button.clicked.connect(self.set_spec_file)
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
        if 'diffractometer' in conf_map:
            diff = str(conf_map['diffractometer']).replace(" ", "")
            self.diffractometer.setText(diff)
        if 'specfile' in conf_map:
            specfile = conf_map['specfile']
            if os.path.isfile(specfile):
                self.spec_file_button.setStyleSheet("Text-align:left")
                self.spec_file_button.setText(specfile)
            else:
                msg_window(f'The specfile file {specfile} in config file does not exist')

        if self.add_config:
            self.extended.load_tab(conf_map)


    def set_spec_file(self):
        """
        Calls selection dialog. The selected spec file is parsed.
        The specfile is saved in config.
        Parameters
        ----------
        none
        Returns
        -------
        noting
        """
        specfile = select_file(os.getcwd())
        if specfile is not None:
            self.spec_file_button.setStyleSheet("Text-align:left")
            self.spec_file_button.setText(specfile)
            if self.add_config:
                self.extended.parse_spec()
        else:
            self.spec_file_button.setText('')

        if self.main_win.is_exp_exists():
            self.save_conf()


    def clear_conf(self):
        self.diffractometer.setText('')
        self.spec_file_button.setText('')
        if self.add_config:
            self.extended.clear_conf()


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
        if len(self.diffractometer.text()) > 0:
            conf_map['diffractometer'] = str(self.diffractometer.text())
        if len(self.spec_file_button.text()) > 0:
            conf_map['specfile'] = str(self.spec_file_button.text())

        if self.add_config:
            conf_map.update(self.extended.get_instr_config())

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

        er_msg = ver.verify('config_instr', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
                return

        ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_instr'))

