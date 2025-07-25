# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This is GUI that allows user to configure and run experiment.
"""

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['select_file',
           'select_dir',
           'msg_window',
           'main']

import sys
import os
import argparse
import shutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import importlib
import ast
import cohere_core.utilities as ut
import cohere_ui.api.common as com
import cohere_ui.api.convertconfig as conv


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


class cdi_gui(QWidget):
    def __init__(self, parent=None, **kwargs):
        """
        Constructor, initializes GUI.
        Parameters
        ----------
        none
        Returns
        -------
        noting
        """
        super(cdi_gui, self).__init__(parent)

        self.loaded = False
        self.beamline = None
        self.id = None
        self.exp_id = None
        self.experiment_dir = None
        self.working_dir = None

        uplayout = QHBoxLayout()
        luplayout = QFormLayout()
        ruplayout = QFormLayout()
        uplayout.addLayout(luplayout)
        uplayout.addLayout(ruplayout)

        self.set_work_dir_button = QPushButton()
        luplayout.addRow("Working Directory", self.set_work_dir_button)
        self.Id_widget = QLineEdit()
        luplayout.addRow("Experiment ID", self.Id_widget)
        self.scan_widget = QLineEdit()
        luplayout.addRow("scan(s)", self.scan_widget)
        self.beamline_widget = QLineEdit()
        ruplayout.addRow("beamline", self.beamline_widget)
        scan_layout = QHBoxLayout()
        self.separate_scans = QCheckBox('separate scans')
        self.separate_scans.setChecked(False)
        scan_layout.addWidget(self.separate_scans)
        self.separate_scan_ranges = QCheckBox('separate scan ranges')
        self.separate_scan_ranges.setChecked(False)
        scan_layout.addWidget(self.separate_scan_ranges)
        self.multipeak = QCheckBox('multi peak')
        self.multipeak.setChecked(False)
        scan_layout.addWidget(self.multipeak)
        luplayout.addRow(scan_layout)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(uplayout)

        self.t = None

        downlayout = QHBoxLayout()
        downlayout.setAlignment(Qt.AlignCenter)
        self.set_exp_button = QPushButton("load experiment")
        self.set_exp_button.setStyleSheet("background-color:rgb(205,178,102)")
        downlayout.addWidget(self.set_exp_button)
        self.create_exp_button = QPushButton('set experiment')
        self.create_exp_button.setStyleSheet("background-color:rgb(120,180,220)")
        downlayout.addWidget(self.create_exp_button)
        self.run_button = QPushButton('run everything', self)
        self.run_button.setStyleSheet("background-color:rgb(175,208,156)")
        downlayout.addWidget(self.run_button)
        self.vbox.addLayout(downlayout)

        spacer = QSpacerItem(0, 5)
        self.vbox.addItem(spacer)

        self.setLayout(self.vbox)
        self.setWindowTitle("Cohere GUI")

        self.set_exp_button.clicked.connect(self.load_experiment)
        self.set_work_dir_button.clicked.connect(self.set_working_dir)
        self.run_button.clicked.connect(self.run_everything)
        self.create_exp_button.clicked.connect(self.set_experiment)
        self.multipeak.stateChanged.connect(self.toggle_multipeak)
        self.separate_scans.stateChanged.connect(self.toggle_separate_scans)
        self.separate_scan_ranges.stateChanged.connect(self.toggle_separate_scan_ranges)


    def set_args(self, args, **kwargs):
        self.args = args
        self.no_verify = kwargs.get('no_verify', False)
        self.debug = kwargs.get('debug', False)


    def run_everything(self):
        """
        Runs everything.py user script in bin directory.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if not self.is_exp_exists():
            msg_window('the experiment has not been created yet')
        elif not self.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
        elif self.t is not None:
            self.t.run_all()


    def reset_window(self):
        self.exp_id = None
        self.experiment_dir = None
        self.working_dir = None
        self.set_work_dir_button.setText('')
        self.Id_widget.setText('')
        self.scan_widget.setText('')
        self.beamline_widget.setText('')
        self.separate_scans.setChecked(False)
        self.separate_scan_ranges.setChecked(False)
        self.multipeak.setChecked(False)

        if self.t is not None:
            self.t.clear_configs()


    def set_working_dir(self):
        """
        It shows the select dialog for user to select working directory. If the selected directory does not exist user will see info message.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.working_dir = select_dir(os.getcwd())
        if self.working_dir is not None:
            self.set_work_dir_button.setStyleSheet("Text-align:left")
            self.set_work_dir_button.setText(self.working_dir)
        else:
            self.set_work_dir_button.setText('')
            msg_window('please select valid working directory')


    def is_exp_exists(self):
        """
        Determines if minimum information for creating the experiment space exists, i.e the working directory and experiment id must be set.
        Resolves the experiment name, and create experiment directory if it does not exist.
        Parameters
        ----------
        none
        Returns
        -------
        boolean
            True if experiment exists, False otherwise
        """
        if self.exp_id is None:
            return False
        if self.working_dir is None:
            return False
        exp_id = str(self.Id_widget.text()).strip()
        scan = str(self.scan_widget.text()).replace(' ','')
        if scan != '':
            exp_id = f'{exp_id}_{scan}'
        if not os.path.exists(ut.join(self.working_dir, exp_id)):
            return False
        return True


    def is_exp_set(self):
        """
        The GUI can be used to load an experiment, and then change the parameters, such id or scan. This function will return True if information in class are the same as in the GUI.
        Parameters
        ----------
        none
        Returns
        -------
        boolean
            True if experiment has been set, False otherwise
        """
        if self.exp_id is None:
            return False
        if self.working_dir is None:
            return False
        if self.id != str(self.Id_widget.text()).strip():
            return False
        return True


    def load_experiment(self):
        """
        It shows a dialog for user to select previously created experiment directory. If no main configuration file is found user will see info message.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.loaded = False
        self.reset_window()
        load_dir = select_dir(os.getcwd())
        if load_dir is None:
            msg_window('please select valid conf directory')
            return
        if not os.path.isfile(ut.join(load_dir, 'conf', 'config')):
            msg_window('missing conf/config file, not experiment directory')
            return

        conf_list = ['config_prep', 'config_data', 'config_rec', 'config_disp', 'config_instr', 'config_mp']
        # set no_verify to True so the configuration is loaded even it's wrong. The user will be able to fix it
        # it is loading the main configuration, so that rec_id is not passed in
        try:
            conf_dicts, converted = com.get_config_maps(load_dir, conf_list, no_verify=True)
        except Exception as e:
            msg_window(str(e))
            return

        self.load_main(conf_dicts['config'])

        if self.t is None:
            try:
                self.t = Tabs(self, self.beamline_widget.text())
                self.vbox.addWidget(self.t)
            except:
                pass
        self.set_experiment(True)
        self.loaded = True
        self.t.load_conf(conf_dicts)

        if not self.is_exp_set():
            return

        if converted:
            self.save_main()
            self.t.save_conf()


    def load_main(self, conf_map):
        """
        It reads 'config' file from the given directory, parses all parameters, verifies, and sets the display in window and class members to parsed values.
        Parameters
        ----------
        load_dir : str
            a directory to load the main configuration from
        Returns
        -------
        nothing
        """
        if 'working_dir' in conf_map:
            working_dir = conf_map['working_dir'].replace(os.sep, '/')
            self.set_work_dir_button.setStyleSheet("Text-align:left")
            self.set_work_dir_button.setText(working_dir)
        if 'experiment_id' in conf_map:
            self.Id_widget.setText(conf_map['experiment_id'])
        if 'scan' in conf_map:
            self.scan_widget.setText(conf_map['scan'].replace(' ',''))
        if 'beamline' in conf_map:
            self.beamline_widget.setText(conf_map['beamline'])
        if 'separate_scans' in conf_map and conf_map['separate_scans']:
            self.separate_scans.setChecked(True)
        if 'separate_scan_ranges' in conf_map and conf_map['separate_scan_ranges']:
            self.separate_scan_ranges.setChecked(True)
        if 'multipeak' in conf_map and conf_map['multipeak']:
            self.multipeak.setChecked(True)


    def assure_experiment_dir(self):
        """
        It creates experiment directory, and experiment configuration directory if they dp not exist.
        Parameters
        ----------
        nothing
        Returns
        -------
        nothing
        """
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        experiment_conf_dir = ut.join(self.experiment_dir, 'conf')
        if not os.path.exists(experiment_conf_dir):
            os.makedirs(experiment_conf_dir)


    def save_main(self):
        # read the configurations from GUI and write to experiment config files
        # save the main config
        conf_map = {}
        conf_map['working_dir'] = str(self.working_dir)
        conf_map['experiment_id'] = self.id
        if len(self.scan_widget.text()) > 0:
            conf_map['scan'] = str(self.scan_widget.text())
        if self.beamline is not None:
            conf_map['beamline'] = self.beamline
        if self.multipeak.isChecked():
            conf_map['multipeak'] = True
        if self.separate_scans.isChecked():
            conf_map['separate_scans'] = True
        if self.separate_scan_ranges.isChecked():
            conf_map['separate_scan_ranges'] = True
        conf_map['converter_ver'] = conv.get_version()
        er_msg = ut.verify('config', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if self.no_verify:
                ut.write_config(conf_map, ut.join(self.experiment_dir, 'conf', 'config'))
        else:
            ut.write_config(conf_map, ut.join(self.experiment_dir, 'conf', 'config'))


    def set_experiment(self, loaded=False):
        """
        Reads the parameters in the window, and sets the experiment to read values, i.e. creates experiment directory,
        and saves all configuration files with parameters from window.

        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        working_dir = self.set_work_dir_button.text().replace(os.sep, '/')
        if len(working_dir) == 0:
            msg_window(
                'The working directory is not defined in config file. Select valid working directory and set experiment')
            return
        elif not os.path.isdir(working_dir):
            msg_window(
                f'The working directory {working_dir} from config file does not exist. Select valid working directory and set experiment')
            self.set_work_dir_button.setText('')
            return
        elif not os.access(working_dir, os.W_OK):
            msg_window(
                f'The working directory {working_dir} is not writable. Select valid working directory and set experiment')
            self.set_work_dir_button.setText('')
            return

        id = str(self.Id_widget.text()).strip()
        if id == '':
            msg_window('id must be entered')
            return

        self.working_dir = working_dir
        self.id = id
        if len(self.scan_widget.text()) > 0:
            self.exp_id = f'{self.id}_{str(self.scan_widget.text()).replace(" ","")}'
        else:
            self.exp_id = self.id
        self.experiment_dir = ut.join(self.working_dir, self.exp_id)
        self.assure_experiment_dir()

        if len(self.beamline_widget.text().strip()) > 0:
            self.beamline = str(self.beamline_widget.text()).strip()
            if not self.t is None:
                self.t.update_beamline(self.beamline)
        else:
            self.beamline = None

        if self.t is None:
            try:
                self.t = Tabs(self, self.beamline_widget.text())
                self.vbox.addWidget(self.t)
            except Exception as e:
                print(str(e))
                pass

        if not loaded:
            self.save_main()
            self.t.save_conf()

        #self.t.notify(**{'experiment_dir': self.experiment_dir})

    def toggle_multipeak(self):
        if self.is_exp_set():
            self.save_main()
        if not self.t is None:
            self.t.toggle_checked(self.multipeak.isChecked(), True)

    def toggle_separate_scans(self):
        if self.is_exp_set():
            self.save_main()
        if not self.t is None:
            self.t.toggle_checked(self.separate_scans.isChecked(), False)

    def toggle_separate_scan_ranges(self):
        if self.is_exp_set():
            self.save_main()
        if not self.t is None:
            self.t.toggle_checked(self.separate_scan_ranges.isChecked(), False)


class Tabs(QTabWidget):
    """
    The main window contains four tabs, each tab holding parameters for different part of processing.
    The tabs are as follows: prep (prepare data), data (format data), rec (reconstruction), disp (visualization).
    This class holds holds the tabs.
    """
    def __init__(self, main_win, beamline, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(Tabs, self).__init__(parent)
        self.main_win = main_win

        if beamline is not None and len(beamline) > 0:
            try:
                self.beam = importlib.import_module(f'cohere_ui.beamlines.{beamline}.beam_tabs')
            except Exception as e:
                print (e)
                msg_window(f'cannot import cohere_ui.beamlines.{beamline} module')
                raise
            self.instr_tab = self.beam.InstrTab()
            self.prep_tab = self.beam.PrepTab()
            self.format_tab = DataTab()
            self.rec_tab = RecTab()
            self.display_tab = DispTab()
            self.tabs = [self.instr_tab, self.prep_tab, self.format_tab, self.rec_tab, self.display_tab]
        else:
            self.format_tab = DataTab()
            self.rec_tab = RecTab()
            self.tabs = [self.format_tab, self.rec_tab]
            self.instr_tab = None
            self.prep_tab = None
            self.display_tab = None
        if self.main_win.multipeak.isChecked():
            self.mp_tab = MpTab()
            self.tabs = self.tabs + [self.mp_tab]

        for tab in self.tabs:
            self.addTab(tab, tab.name)
            tab.init(self, main_win)


    def update_beamline(self, beamline):
        # a case when beamline tab is already set
        if not self.instr_tab is None:
            return
        try:
            self.beam = importlib.import_module(f'cohere_ui.beamlines.{beamline}.beam_tabs')
        except Exception as e:
            print (e)
            msg_window(f'cannot import cohere_ui.beamlines.{beamline} module')
            raise
        self.instr_tab = self.beam.InstrTab()
        self.insertTab(0, self.instr_tab, self.instr_tab.name)
        self.instr_tab.init(self, self.main_win)
        self.prep_tab = self.beam.PrepTab()
        self.insertTab(1, self.prep_tab, self.prep_tab.name)
        self.prep_tab.init(self, self.main_win)
        self.tabs = self.tabs + [self.instr_tab, self.prep_tab]

    def notify(self, **args):
        try:
            self.display_tab.update_tab(**args)
            self.prep_tab.update_tab(**args)
        except:
            pass


    def clear_configs(self):
        for tab in self.tabs:
            tab.clear_conf()


    def run_all(self):
        for tab in self.tabs:
            tab.run_tab()


    def run_prep(self):
        import cohere_ui.beamline_preprocess as prep

        # this line is passing all parameters from command line to prep script. 
        # if there are other parameters, one can add some code here        
        try:
            prep.handle_prep(self.main_win.experiment_dir, no_verify=self.main_win.no_verify)
        except ValueError as e:
            msg_window(str(e))
        except KeyError as e:
            msg_window(str(e))


    def run_viz(self):
        import cohere_ui.beamline_visualization as dp

        try:
            dp.handle_visualization(self.main_win.experiment_dir, no_verify=self.main_win.no_verify)
        except ValueError as e:
            msg_window(str(e))
        except KeyError as e:
            msg_window(str(e))


    def load_conf(self, conf_dirs):
        for tab in self.tabs:
            if tab.conf_name in conf_dirs.keys():
                tab.load_tab(conf_dirs[tab.conf_name])


    def save_conf(self):
        for tab in self.tabs:
            tab.save_conf()


    def toggle_checked(self, is_checked, is_multipeak):
        if is_multipeak:
            if is_checked:
                self.mp_tab = MpTab()
                self.addTab(self.mp_tab, self.mp_tab.name)
                self.mp_tab.init(self, self.main_win)
                self.tabs = self.tabs + [self.mp_tab]
            else:
                self.removeTab(self.count()-1)
                self.tabs.remove(self.mp_tab)
                self.mp_tab = None

        # change the Instrument tab if present
        if not self.instr_tab is None:
            self.instr_tab.toggle_config()


class DataTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(DataTab, self).__init__(parent)
        self.name = 'Data'
        self.conf_name = 'config_data'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'data' tab.
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
        self.alien_alg = QComboBox()
        self.alien_alg.addItem("none")
        self.alien_alg.addItem("block aliens")
        self.alien_alg.addItem("alien file")
        self.alien_alg.addItem("AutoAlien1")
        layout.addRow("alien algorithm", self.alien_alg)
        sub_layout = QFormLayout()
        self.set_alien_layout(sub_layout)
        layout.addRow(sub_layout)
        self.auto_intensity_threshold = QCheckBox('auto intensity threshold')
        self.auto_intensity_threshold.setChecked(False)
        layout.addWidget(self.auto_intensity_threshold)
        self.intensity_threshold = QLineEdit()
        layout.addRow("Intensity Threshold", self.intensity_threshold)
        self.shift = QLineEdit()
        layout.addRow("shift", self.shift)
        self.crop_pad = QLineEdit()
        layout.addRow("crop, pad", self.crop_pad)
        self.binning = QLineEdit()
        layout.addRow("binning", self.binning)
        self.no_center_max = QCheckBox('not center max')
        self.no_center_max.setChecked(False)
        layout.addWidget(self.no_center_max)
        cmd_layout = QHBoxLayout()
        self.set_data_conf_from_button = QPushButton("Load data conf from")
        self.set_data_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.config_data_button = QPushButton('format data', self)
        self.config_data_button.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_data_conf_from_button)
        cmd_layout.addWidget(self.config_data_button)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.alien_alg.currentIndexChanged.connect(lambda: self.set_alien_layout(sub_layout))
        # this will create config_data file and run data script
        # to generate data ready for reconstruction
        self.config_data_button.clicked.connect(self.run_tab)
        self.set_data_conf_from_button.clicked.connect(self.load_data_conf)


    def clear_conf(self):
        self.alien_alg.setCurrentIndex(0)
        self.intensity_threshold.setText('')
        self.binning.setText('')
        self.shift.setText('')
        self.crop_pad.setText('')
        self.auto_intensity_threshold.setChecked(False)
        self.no_center_max.setChecked(False)


    def load_tab(self, conf_map):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf_map : dict
            configuration (config_data)
        Returns
        -------
        nothing
        """
        if 'alien_alg' not in conf_map:
            conf_map['alien_alg'] = 'random'
        if conf_map['alien_alg'] == 'random':
            self.alien_alg.setCurrentIndex(0)
        elif conf_map['alien_alg'] == 'block_aliens':
            self.alien_alg.setCurrentIndex(1)
            if 'aliens' in conf_map:
                self.aliens.setText(str(conf_map['aliens']).replace(" ", ""))
        elif conf_map['alien_alg'] == 'alien_file':
            self.alien_alg.setCurrentIndex(2)
            if 'alien_file' in conf_map:
                self.alien_file.setText(str(conf_map['alien_file']).replace(" ", ""))
        elif conf_map['alien_alg'] == 'AutoAlien1':
            self.alien_alg.setCurrentIndex(3)
            if 'AA1_size_threshold' in conf_map:
                self.AA1_size_threshold.setText(str(conf_map['AA1_size_threshold']).replace(" ", ""))
            if 'AA1_asym_threshold' in conf_map:
                self.AA1_asym_threshold.setText(str(conf_map['AA1_asym_threshold']).replace(" ", ""))
            if 'AA1_min_pts' in conf_map:
                self.AA1_min_pts.setText(str(conf_map['AA1_min_pts']).replace(" ", ""))
            if 'AA1_eps' in conf_map:
                self.AA1_eps.setText(str(conf_map['AA1_eps']).replace(" ", ""))
            if 'AA1_amp_threshold' in conf_map:
                self.AA1_amp_threshold.setText(str(conf_map['AA1_amp_threshold']).replace(" ", ""))
            if 'AA1_save_arrs' in conf_map:
                self.AA1_save_arrs.setChecked(conf_map['AA1_save_arrs'])
            else:
                self.AA1_save_arrs.setChecked(False)
            if 'AA1_expandcleanedsigma' in conf_map:
                self.AA1_expandcleanedsigma.setText(str(conf_map['AA1_expandcleanedsigma']).replace(" ", ""))
        self.auto_intensity_threshold.setChecked('auto_intensity_threshold' in conf_map and conf_map['auto_intensity_threshold'])
        if 'intensity_threshold' in conf_map:
            self.intensity_threshold.setText(str(conf_map['intensity_threshold']).replace(" ", ""))
        if 'binning' in conf_map:
            self.binning.setText(str(conf_map['binning']).replace(" ", ""))
        if 'shift' in conf_map:
            self.shift.setText(str(conf_map['shift']).replace(" ", ""))
        if 'crop_pad' in conf_map:
            self.crop_pad.setText(str(conf_map['crop_pad']).replace(" ", ""))
        if 'no_center_max' in conf_map and conf_map['no_center_max']:
            self.no_center_max.setChecked(True)
        else:
            self.no_center_max.setChecked(False)


    def get_data_config(self):
        """
        It reads parameters related to formatting data from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}

        if self.alien_alg.currentIndex() == 1:
            conf_map['alien_alg'] = 'block_aliens'
            if len(self.aliens.text()) > 0:
                conf_map['aliens'] = str(self.aliens.text()).replace(os.linesep, '')
        if self.alien_alg.currentIndex() == 2:
            conf_map['alien_alg'] = 'alien_file'
            if len(self.alien_file.text()) > 0:
                conf_map['alien_file'] = str(self.alien_file.text())
        elif self.alien_alg.currentIndex() == 3:
            conf_map['alien_alg'] = 'AutoAlien1'
            if len(self.AA1_size_threshold.text()) > 0:
                conf_map['AA1_size_threshold'] = ast.literal_eval(str(self.AA1_size_threshold.text()))
            if len(self.AA1_asym_threshold.text()) > 0:
                conf_map['AA1_asym_threshold'] = ast.literal_eval(str(self.AA1_asym_threshold.text()))
            if len(self.AA1_min_pts.text()) > 0:
                conf_map['AA1_min_pts'] = ast.literal_eval(str(self.AA1_min_pts.text()))
            if len(self.AA1_eps.text()) > 0:
                conf_map['AA1_eps'] = ast.literal_eval(str(self.AA1_eps.text()))
            if len(self.AA1_amp_threshold.text()) > 0:
                conf_map['AA1_amp_threshold'] = ast.literal_eval(str(self.AA1_amp_threshold.text()))
            if self.AA1_save_arrs.isChecked():
                conf_map['AA1_save_arrs'] = True
            if len(self.AA1_expandcleanedsigma.text()) > 0:
                conf_map['AA1_expandcleanedsigma'] = ast.literal_eval(str(self.AA1_expandcleanedsigma.text()))

        if len(self.intensity_threshold.text()) > 0:
            conf_map['intensity_threshold'] = ast.literal_eval(str(self.intensity_threshold.text()))
        if len(self.binning.text()) > 0:
            conf_map['binning'] = ast.literal_eval(str(self.binning.text()).replace(os.linesep, ''))
        if len(self.shift.text()) > 0:
            conf_map['shift'] = ast.literal_eval(str(self.shift.text()).replace(os.linesep, ''))
        if len(self.crop_pad.text()) > 0:
            conf_map['crop_pad'] = ast.literal_eval(str(self.crop_pad.text()).replace(os.linesep, ''))
        if self.auto_intensity_threshold.isChecked():
            conf_map['auto_intensity_threshold'] = True
        if self.no_center_max.isChecked():
            conf_map['no_center_max'] = True

        return conf_map


    def set_alien_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        if self.alien_alg.currentIndex() == 1:
            self.aliens = QLineEdit()
            layout.addRow("aliens", self.aliens)
        elif self.alien_alg.currentIndex() == 2:
            self.alien_file = QPushButton()
            layout.addRow("alien file", self.alien_file)
            self.alien_file.clicked.connect(self.set_alien_file)
        elif self.alien_alg.currentIndex() == 3:
            self.AA1_size_threshold = QLineEdit()
            layout.addRow("relative size threshold", self.AA1_size_threshold)
            self.AA1_asym_threshold = QLineEdit()
            layout.addRow("average asymmetry threshold", self.AA1_asym_threshold)
            self.AA1_min_pts = QLineEdit()
            layout.addRow("min pts in cluster", self.AA1_min_pts)
            self.AA1_eps = QLineEdit()
            layout.addRow("cluster alg eps", self.AA1_eps)
            self.AA1_amp_threshold = QLineEdit()
            layout.addRow("alien alg amp threshold", self.AA1_amp_threshold)
            self.AA1_save_arrs = QCheckBox()
            layout.addRow("save analysis arrs", self.AA1_save_arrs)
            self.AA1_save_arrs.setChecked(False)
            self.AA1_expandcleanedsigma = QLineEdit()
            layout.addRow("expand cleaned sigma", self.AA1_expandcleanedsigma)
            self.AA1_default_button = QPushButton('set AutoAlien1 parameters to defaults', self)
            layout.addWidget(self.AA1_default_button)

            self.AA1_default_button.clicked.connect(self.set_AA1_defaults)


    def set_AA1_defaults(self):
        """
        Sets the AutoAlien1 parameters in the data tab to hardcoded defaults.
        """
        self.AA1_size_threshold.setText('0.01')
        self.AA1_asym_threshold.setText('1.75')
        self.AA1_min_pts.setText('5')
        self.AA1_eps.setText('1.1')
        self.AA1_save_arrs.setText('False')
        self.AA1_amp_threshold.setText('6.0')


    def set_alien_file(self):
        """
        It display a select dialog for user to select an alien file.
        """
        self.alien_filename = select_file(os.getcwd())
        if self.alien_filename is not None:
            self.alien_file.setStyleSheet("Text-align:left")
            self.alien_file.setText(self.alien_filename)
        else:
            self.alien_file.setText('')


    def run_tab(self):
        """
        Reads the parameters needed by format data script. Saves the config_data configuration file with parameters from the window and runs the format script.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        import cohere_ui.standard_preprocess as run_dt

        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
            return
        elif not self.main_win.is_exp_set():
            msg_window('the experiment has changed, press "set experiment" button')
            return
        else:
            found_file = False
            for p, d, f in os.walk(self.main_win.experiment_dir):
                if 'prep_data.tif' in f:
                    found_file = True
                    break
            if found_file:
                conf_map = self.get_data_config()
                if len(conf_map) > 0:
                    # verify that data configuration is ok
                    er_msg = ut.verify('config_data', conf_map)
                    if len(er_msg) > 0:
                        msg_window(er_msg)
                        if not self.main_win.no_verify:
                            return
                    ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_data'))
                try:
                    run_dt.format_data(self.main_win.experiment_dir, no_verify=self.main_win.no_verify)
                except Exception as e:
                    msg_window(str(e))
                    return
            else:
                msg_window('Run data preparation in previous tab to activate this function')
                return

        # reload the window if auto_intensity_threshold is set
        if self.auto_intensity_threshold.isChecked():
            data_map = ut.read_config(ut.join(self.main_win.experiment_dir, 'conf', 'config_data'))
            self.load_tab(data_map)


    def save_conf(self):
        # save data config
        conf_map = self.get_data_config()
        if len(conf_map) > 0:
            er_msg = ut.verify('config_data', conf_map)
            if len(er_msg) > 0:
                msg_window(er_msg)
                if not self.main_win.no_verify:
                    return
            ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_data'))


    def load_data_conf(self):
        """
        It display a select dialog for user to select a configuration file. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        data_file = select_file(os.getcwd())
        if data_file is not None:
            conf_map = ut.read_config(data_file.replace(os.sep, '/'))
            self.load_tab(conf_map)
        else:
            msg_window('please select valid data config file')


class RecTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(RecTab, self).__init__(parent)
        self.name = 'Reconstruction'
        self.conf_name = 'config_rec'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'reconstruction' tab.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.tabs = tabs
        self.main_win = main_window
        self.old_rec_id = ''

        layout = QVBoxLayout()
        ulayout = QFormLayout()
        mlayout = QHBoxLayout()

        self.init_guess = QComboBox()
        self.init_guess.InsertAtBottom
        self.init_guess.addItem("random")
        self.init_guess.addItem("continue")
        self.init_guess.addItem("AI algorithm")
        ulayout.addRow("initial guess", self.init_guess)
        sub_layout = QFormLayout()
        self.set_init_guess_layout(sub_layout)
        ulayout.addRow(sub_layout)

        self.add_conf_button = QPushButton('add configuration', self)
        ulayout.addWidget(self.add_conf_button)
        self.rec_id = QComboBox()
        self.rec_id.InsertAtBottom
        self.rec_id.addItem("main")
        ulayout.addWidget(self.rec_id)
        self.rec_id.hide()
        self.proc = QComboBox()
        self.proc.addItem("auto")
        if sys.platform != 'darwin':
            self.proc.addItem("cp")
        self.proc.addItem("np")
        self.proc.addItem("torch")
        ulayout.addRow("processor type", self.proc)
        self.device = QLineEdit()
        ulayout.addRow("device(s)", self.device)
        self.reconstructions = QLineEdit()
        ulayout.addRow("number of reconstructions", self.reconstructions)
        self.alg_seq = QLineEdit()
        ulayout.addRow("algorithm sequence", self.alg_seq)
        # TODO add logic to show this only if HIO is in sequence
        self.hio_beta = QLineEdit()
        ulayout.addRow("HIO beta", self.hio_beta)
        self.raar_beta = QLineEdit()
        ulayout.addRow("RAAR beta", self.raar_beta)
        self.initial_support_area = QLineEdit()
        ulayout.addRow("initial support area", self.initial_support_area)
        self.rec_default_button = QPushButton('set to defaults', self)
        ulayout.addWidget(self.rec_default_button)

        self.features = Features(self, mlayout)

        llayout = QHBoxLayout()
        self.set_rec_conf_from_button = QPushButton("Load rec conf from")
        self.set_rec_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.config_rec_button = QPushButton('run reconstruction', self)
        self.config_rec_button.setStyleSheet("background-color:rgb(175,208,156)")
        llayout.addWidget(self.set_rec_conf_from_button)
        llayout.addWidget(self.config_rec_button)

        spacer = QSpacerItem(0, 3)
        llayout.addItem(spacer)

        layout.addLayout(ulayout)
        layout.addLayout(mlayout)
        layout.addLayout(llayout)

        self.setAutoFillBackground(True)
        self.setLayout(layout)

        self.config_rec_button.clicked.connect(self.run_tab)
        self.init_guess.currentIndexChanged.connect(lambda: self.set_init_guess_layout(sub_layout))
        self.rec_default_button.clicked.connect(self.set_defaults)
        self.add_conf_button.clicked.connect(self.add_rec_conf)
        self.rec_id.currentIndexChanged.connect(self.toggle_conf)
        self.set_rec_conf_from_button.clicked.connect(self.load_rec_conf_dir)


    def load_tab(self, conf_map, update_rec_choice=True):
        if 'init_guess' not in conf_map:
            conf_map['init_guess'] = 'random'
        if conf_map['init_guess'] == 'random':
            self.init_guess.setCurrentIndex(0)
        elif conf_map['init_guess'] == 'continue':
            self.init_guess.setCurrentIndex(1)
            if 'continue_dir' in conf_map:
                self.cont_dir_button.setText(str(conf_map['continue_dir'].replace(os.sep, '/')).replace(" ", ""))
        elif conf_map['init_guess'] == 'AI_guess':
            self.init_guess.setCurrentIndex(2)
            if 'AI_trained_model' in conf_map:
                self.AI_trained_model.setText(str(conf_map['AI_trained_model'].replace(os.sep, '/')).replace(" ", ""))
                self.AI_trained_model.setStyleSheet("Text-align:left")

        # this will update the configuration choices by reading configuration files names
        # do not update when doing toggle
        self.rec_ids = []
        if update_rec_choice:
            self.update_rec_configs_choice()

        if 'processing' in conf_map:
            self.proc.setCurrentText(str(conf_map['processing']))
        if 'device' in conf_map:
            self.device.setText(str(conf_map['device']).replace(" ", ""))
        if 'reconstructions' in conf_map:
            self.reconstructions.setText(str(conf_map['reconstructions']).replace(" ", ""))
        if 'algorithm_sequence' in conf_map:
            self.alg_seq.setText(str(conf_map['algorithm_sequence']))
        if 'hio_beta' in conf_map:
            self.hio_beta.setText(str(conf_map['hio_beta']).replace(" ", ""))
        if 'raar_beta' in conf_map:
            self.raar_beta.setText(str(conf_map['raar_beta']).replace(" ", ""))
        if 'initial_support_area' in conf_map:
            self.initial_support_area.setText(str(conf_map['initial_support_area']).replace(" ", ""))

        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].init_config(conf_map)

        # self.notify()


    def clear_conf(self):
        self.init_guess.setCurrentIndex(0)
        nu_to_remove = self.rec_id.count() - 1
        for _ in range(nu_to_remove):
            self.rec_id.removeItem(1)
        self.old_rec_id = ''
        self.device.setText('')
        self.proc.setCurrentIndex(0)
        self.reconstructions.setText('')
        self.alg_seq.setText('')
        self.hio_beta.setText('')
        self.raar_beta.setText('')
        self.initial_support_area.setText('')
        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].active.setChecked(False)


    def get_rec_config(self):
        """
        It reads parameters related to reconstruction from the window and adds them to dictionary.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}
        if len(self.reconstructions.text()) > 0:
            try:
                conf_map['reconstructions'] = ast.literal_eval(self.reconstructions.text())
            except:
                msg_window('reconstructions parameter should be int')
                return {}
        if len(self.proc.currentText()) > 0:
            conf_map['processing'] = str(self.proc.currentText())
        if len(self.device.text()) > 0:
            try:
                d = str(self.device.text()).replace(os.linesep,'')
                if d == 'all':
                    conf_map['device'] = d
                else:
                    conf_map['device'] = ast.literal_eval(d)
            except:
                msg_window('device parameter should be "all" or a list of int or dict')
                return {}
        if len(self.alg_seq.text()) > 0:
            conf_map['algorithm_sequence'] = str(self.alg_seq.text()).strip()
        if len(self.hio_beta.text()) > 0:
            try:
                conf_map['hio_beta'] = ast.literal_eval(str(self.hio_beta.text()))
            except:
                msg_window('hio_beta parameter should be float')
                return {}
        if len(self.raar_beta.text()) > 0:
            try:
                conf_map['raar_beta'] = ast.literal_eval(str(self.raar_beta.text()))
            except:
                msg_window('raar_beta parameter should be float')
                return {}
        if len(self.initial_support_area.text()) > 0:
            try:
                conf_map['initial_support_area'] = ast.literal_eval(str(self.initial_support_area.text()).replace(os.linesep,''))
            except:
                msg_window('initial_support_area parameter should be a list of floats')
                return {}
        if self.init_guess.currentIndex() == 1:
            conf_map['init_guess'] = 'continue'
            if len(self.cont_dir_button.text().strip()) > 0:
                conf_map['continue_dir'] = str(self.cont_dir_button.text()).replace(os.sep, '/').strip()
        elif self.init_guess.currentIndex() == 2:
            conf_map['init_guess'] = 'AI_guess'
            if len(self.AI_trained_model.text()) > 0:
                conf_map['AI_trained_model'] = str(self.AI_trained_model.text()).replace(os.sep, '/').strip()
        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].add_config(conf_map)

        return conf_map


    def save_conf(self):
        conf_map = self.get_rec_config()
        if len(conf_map) == 0:
            return
        er_msg = ut.verify('config_rec', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
                return

        ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_rec'))


    def set_init_guess_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        if self.init_guess.currentIndex() == 1:
            self.cont_dir_button = QPushButton()
            layout.addRow("continue directory", self.cont_dir_button)
            self.cont_dir_button.clicked.connect(self.set_cont_dir)
        elif self.init_guess.currentIndex() == 2:
            self.AI_trained_model = QPushButton()
            layout.addRow("AI trained model file", self.AI_trained_model)
            self.AI_trained_model.clicked.connect(self.set_aitm_file)


    def set_cont_dir(self):
        """
        It display a select dialog for user to select a directory with raw data file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        cont_dir = select_dir(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if cont_dir is not None:
            self.cont_dir_button.setStyleSheet("Text-align:left")
            self.cont_dir_button.setText(cont_dir)
        else:
            self.cont_dir_button.setText('')


    def set_aitm_file(self):
        AI_trained_model = select_file(os.getcwd().replace(os.sep, '/')).replace(os.sep, '/')
        if AI_trained_model is not None:
            self.AI_trained_model.setStyleSheet("Text-align:left")
            self.AI_trained_model.setText(AI_trained_model)
        else:
            self.AI_trained_model.setText('')


    def add_rec_conf(self):
        id, ok = QInputDialog.getText(self, '', "enter configuration id")
        if id in self.rec_ids:
            msg_window(f'the {id} is alredy used')
            return
        if ok and len(id) > 0:
            if len(self.rec_ids) > 1:
                self.rec_id.addItem(id)
            else:
                self.rec_id.show()
                self.rec_id.addItem(id)
        else:
            return

        # copy the config_rec into <id>_config_rec

        conf_file = ut.join(self.main_win.experiment_dir, 'conf', 'config_rec')
        new_conf_file = ut.join(self.main_win.experiment_dir, 'conf', f'config_rec_{id}')
        shutil.copyfile(conf_file, new_conf_file)
        self.rec_id.setCurrentIndex(self.rec_id.count() - 1)


    def toggle_conf(self):
        """
        Invoked when the configuration to use in the reconstruction was changed. This will bring the parameters from
        the previous config to be saved, and the new ones retrieved and showed in window.
        Parameters
        ----------
        layout : QFormLayout
            a layout to add the continue dir

        Returns
        -------
        nothing
        """
        if self.main_win.experiment_dir is None:
            return
        # save the configuration file before updating the incoming config
        if self.old_rec_id == '':
            conf_file = 'config_rec'
        else:
            conf_file =  f'config_rec_{self.old_rec_id}'

        conf_map = self.get_rec_config()
        if len(conf_map) == 0:
            return
        conf_dir = ut.join(self.main_win.experiment_dir, 'conf')

        ut.write_config(conf_map, ut.join(conf_dir, conf_file))
        if str(self.rec_id.currentText()) == 'main':
            self.old_rec_id = ''
        else:
            self.old_rec_id = str(self.rec_id.currentText())
        # if a config file corresponding to the rec id exists, load it
        # otherwise read from base configuration and load
        if self.old_rec_id == '':
            conf_file = ut.join(conf_dir, 'config_rec')
        else:
            conf_file = ut.join(conf_dir, f'config_rec_{self.old_rec_id}')

        conf_map = ut.read_config(conf_file)
        if conf_map is None:
            msg_window(f'please check configuration file {conf_file}')
            return
        self.load_tab(conf_map, False)
        self.notify()


    def load_rec_conf_dir(self):
        """
        It displays a select dialog for user to select a configuration file. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        rec_file = select_file(os.getcwd())
        if rec_file is not None:
            conf_map = ut.read_config(rec_file.replace(os.sep, '/'))
            if conf_map is None:
                msg_window(f'please check configuration file {rec_file}')
                return

            self.load_tab(conf_map)
        else:
            msg_window('please select valid rec config file')


    def run_tab(self):
        """
        Reads the parameters needed by reconstruction script. Saves the config_rec configuration file with parameters from the window and runs the reconstruction script.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        import cohere_ui.run_reconstruction as run_rc

        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
            return

        if not self.main_win.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
            return

        found_file = False
        for p, d, f in os.walk(self.main_win.experiment_dir):
            if 'data.tif' in f:
                found_file = True
                break
            if 'data.npy' in f:
                found_file = True
                break
        if found_file:
            # find out which configuration should be saved
            if self.old_rec_id == '':
                conf_file = 'config_rec'
                rec_id = None
            else:
                conf_file =  'config_rec_' + self.old_rec_id
                rec_id = self.old_rec_id

            conf_map = self.get_rec_config()
            if len(conf_map) == 0:
                return

            # verify that reconstruction configuration is ok
            er_msg = ut.verify('config_rec', conf_map)
            if len(er_msg) > 0:
                msg_window(er_msg)
                if not self.main_win.no_verify:
                    return
            ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', conf_file))
            try:
                run_rc.manage_reconstruction(self.main_win.experiment_dir,
                                         rec_id=rec_id,
                                         no_verify=self.main_win.no_verify,
                                         debug=self.main_win.debug)
            except Exception as e:
                msg_window(str(e))
                return
            self.notify(rec_conf_map=conf_map)
        else:
            msg_window('Please, run format data in previous tab to activate this function')


    def set_defaults(self):
        """
        Sets the basic parameters in the reconstruction tab main part to hardcoded defaults.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        if self.main_win.working_dir is None or self.main_win.id is None or \
                        len(self.main_win.working_dir) == 0 or len(self.main_win.id) == 0:
            msg_window('Working Directory or Reconstruction ID not configured')
        else:
            self.reconstructions.setText('1')
            self.proc.setCurrentIndex(0)
            self.device.setText('[0]')
            self.alg_seq.setText('3*(20*ER+180*HIO)+20*ER')
            self.hio_beta.setText('.9')
            self.raar_beta.setText('.45')
            self.initial_support_area.setText('[0.5, 0.5, 0.5]')


    def update_rec_configs_choice(self):
        """
        Looks for alternate reconstruction configurations, and updates window with that information.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        # this will update the configuration choices in reconstruction tab
        # fill out the rec_id choice bar by reading configuration files names
        if not self.main_win.is_exp_set():
            return
        for file in os.listdir(ut.join(self.main_win.experiment_dir, 'conf')):
            if file.startswith('config_rec_'):
                self.rec_ids.append(file[len('config_rec_') : len(file)])
        if len(self.rec_ids) > 0:
            self.rec_id.addItems(self.rec_ids)
            self.rec_id.show()


    def notify(self, **kwargs):
        self.tabs.display_tab.update_tab(**kwargs)


class Feature(object):
    """
    This is a parent class to concrete feature classes.
    """
    def __init__(self):
        """
        Constructor, each feature object contains QWidget.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.stack = QWidget()


    def stackUI(self, item, feats):
        """
        Used by all sub-classes (features) when initialized.
        Parameters
        ----------
        item : item from QListWidget
            item represents a feature
        feats : Features object
            Features object is a composition of features
        Returns
        -------
        nothing
        """
        layout = QFormLayout()
        self.active = QCheckBox("active")
        layout.addWidget(self.active)
        self.toggle(layout, item, feats)
        self.stack.setLayout(layout)
        self.active.stateChanged.connect(lambda: self.toggle(layout, item, feats))


    def toggle(self, layout, item, feats):
        """
        Used by sub-classes (features) when a feature is activated or deactivated.
        Parameters
        ----------
        item : item from QListWidget
            item represents a feature
        feats : Features object
            Features object is a composition of features
        Returns
        -------
        nothing
        """
        if self.active.isChecked():
            self.fill_active(layout)

            self.default_button = QPushButton('set to defaults', feats)
            layout.addWidget(self.default_button)
            self.default_button.clicked.connect(self.rec_default)

            item.setForeground(QColor('black'));
        else:
            self.clear_params(layout, item)


    def clear_params(self, layout, item):
        for i in reversed(range(1, layout.count())):
            layout.itemAt(i).widget().setParent(None)
        item.setForeground(QColor('grey'));


    def fill_active(self, layout):
        """
        This function is overriden in concrete class. It displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        pass


    def rec_default(self):
        """
        This function is overriden in concrete class. It sets feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        pass


    def add_config(self, conf_map):
        """
        This function calls all of the subclasses to add feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if self.active.isChecked():
            self.add_feat_conf(conf_map)


    def add_feat_conf(self, conf_map):
        """
        This function is overriden in concrete class. It adds feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        pass


    def init_config(self, conf_map):
        """
        This function is overriden in concrete class. It sets feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        pass


class GA(Feature):
    """
    This class encapsulates GA feature.
    """
    def __init__(self):
        super(GA, self).__init__()
        self.id = 'GA'

    # override setting the active to set it False
    def stackUI(self, item, feats):
        super(GA, self).stackUI(item, feats)


    def init_config(self, conf_map):
        """
        This function sets GA feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'ga_generations' in conf_map:
            gens = conf_map['ga_generations']
            self.active.setChecked(True)
            self.generations.setText(str(gens).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        self.ga_fast.setChecked(conf_map.get('ga_fast', False))
        if 'ga_metrics' in conf_map:
            self.metrics.setText(str(conf_map['ga_metrics']).replace(" ", ""))
        else:
            self.metrics.setText('')
        if 'ga_breed_modes' in conf_map:
            self.breed_modes.setText(str(conf_map['ga_breed_modes']).replace(" ", ""))
        else:
            self.breed_modes.setText('')
        if 'ga_cullings' in conf_map:
            self.removes.setText(str(conf_map['ga_cullings']).replace(" ", ""))
        else:
            self.removes.setText('')
        if 'ga_sw_thresholds' in conf_map:
            self.ga_sw_thresholds.setText(str(conf_map['ga_sw_thresholds']).replace(" ", ""))
        else:
            self.ga_sw_thresholds.setText('')
        if 'ga_sw_gauss_sigmas' in conf_map:
            self.ga_sw_gauss_sigmas.setText(str(conf_map['ga_sw_gauss_sigmas']).replace(" ", ""))
        else:
            self.ga_sw_gauss_sigmas.setText('')
        if 'ga_lpf_sigmas' in conf_map:
            self.lr_sigmas.setText(str(conf_map['ga_lpf_sigmas']).replace(" ", ""))
        else:
            self.lr_sigmas.setText('')
        if 'ga_gen_pc_start' in conf_map:
            self.gen_pc_start.setText(str(conf_map['ga_gen_pc_start']).replace(" ", ""))
        else:
            self.gen_pc_start.setText('')


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.ga_fast = QCheckBox("fast processing, size limited")
        self.ga_fast.setChecked(False)
        layout.addWidget(self.ga_fast)
        self.generations = QLineEdit()
        layout.addRow("generations", self.generations)
        self.metrics = QLineEdit()
        layout.addRow("fitness metrics", self.metrics)
        self.breed_modes = QLineEdit()
        layout.addRow("breed modes", self.breed_modes)
        self.removes = QLineEdit()
        layout.addRow("cullings", self.removes)
        self.ga_sw_thresholds = QLineEdit()
        layout.addRow("after breed support thresholds", self.ga_sw_thresholds)
        self.ga_sw_gauss_sigmas = QLineEdit()
        layout.addRow("after breed shrink wrap sigmas", self.ga_sw_gauss_sigmas)
        self.lr_sigmas = QLineEdit()
        layout.addRow("low resolution sigmas", self.lr_sigmas)
        self.gen_pc_start = QLineEdit()
        layout.addRow("gen to start pcdi", self.gen_pc_start)


    def rec_default(self):
        """
        This function sets GA feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.generations.setText('3')
        self.metrics.setText('["chi"]')
        self.breed_modes.setText('["sqrt_ab"]')
        self.ga_sw_thresholds.setText('[.1]')
        self.ga_sw_gauss_sigmas.setText('[1.0]')
        self.gen_pc_start.setText('3')
        self.active.setChecked(True)


    def add_feat_conf(self, conf_map):
        """
        This function adds GA feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if self.ga_fast.isChecked():
            conf_map['ga_fast'] = True
        if len(self.generations.text()) > 0:
            conf_map['ga_generations'] = ast.literal_eval(str(self.generations.text()))
        if len(self.metrics.text()) > 0:
         conf_map['ga_metrics'] = ast.literal_eval(str(self.metrics.text()).replace(os.linesep,''))
        if len(self.breed_modes.text()) > 0:
          conf_map['ga_breed_modes'] = ast.literal_eval(str(self.breed_modes.text()).replace(os.linesep,''))
        if len(self.removes.text()) > 0:
           conf_map['ga_cullings'] = ast.literal_eval(str(self.removes.text()).replace(os.linesep,''))
        if len(self.ga_sw_thresholds.text()) > 0:
            conf_map['ga_sw_thresholds'] = ast.literal_eval(str(self.ga_sw_thresholds.text()).replace(os.linesep,''))
        if len(self.ga_sw_gauss_sigmas.text()) > 0:
            conf_map['ga_sw_gauss_sigmas'] = ast.literal_eval(str(self.ga_sw_gauss_sigmas.text()).replace(os.linesep,''))
        if len(self.lr_sigmas.text()) > 0:
            conf_map['ga_lpf_sigmas'] = ast.literal_eval(str(self.lr_sigmas.text()).replace(os.linesep,''))
        if len(self.gen_pc_start.text()) > 0:
            conf_map['ga_gen_pc_start'] = ast.literal_eval(str(self.gen_pc_start.text()))


class low_resolution(Feature):
    """
    This class encapsulates low resolution feature.
    """
    def __init__(self):
        super(low_resolution, self).__init__()
        self.id = 'low resolution'


    def init_config(self, conf_map):
        """
        This function sets low resolution feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'lowpass_filter_trigger' in conf_map:
            triggers = conf_map['lowpass_filter_trigger']
            self.active.setChecked(True)
            self.lpf_triggers.setText(str(triggers).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'lowpass_filter_sw_threshold' in conf_map:
            self.lpf_sw_threshold.setText(str(conf_map['lowpass_filter_sw_threshold']).replace(" ", ""))
        else:
            self.lpf_sw_threshold.setText('')
        if 'lowpass_filter_range' in conf_map:
            self.lpf_range.setText(str(conf_map['lowpass_filter_range']).replace(" ", ""))
        else:
            self.lpf_range.setText('')


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.lpf_triggers = QLineEdit()
        layout.addRow("lowpass filter triggers", self.lpf_triggers)
        self.lpf_triggers.setToolTip('suggested trigger: [0, 1, <half iteration number>]')
        self.lpf_sw_threshold = QLineEdit()
        layout.addRow("shrink wrap threshold", self.lpf_sw_threshold)
        self.lpf_range = QLineEdit()
        layout.addRow("lowpass filter range", self.lpf_range)


    def rec_default(self):
        """
        This function sets low resolution feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.lpf_triggers.setText('[0, 1, 320]')
        self.lpf_sw_threshold.setText('.1')
        self.lpf_range.setText('[.7]')


    def add_feat_conf(self, conf_map):
        """
        This function adds low resolution feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.lpf_triggers.text()) > 0:
            conf_map['lowpass_filter_trigger'] = ast.literal_eval(str(self.lpf_triggers.text()).replace(os.linesep, ''))
        if len(self.lpf_sw_threshold.text()) > 0:
            conf_map['lowpass_filter_sw_threshold'] = ast.literal_eval(str(self.lpf_sw_threshold.text()).replace(os.linesep, ''))
        if len(self.lpf_range.text()) > 0:
            conf_map['lowpass_filter_range'] = ast.literal_eval(str(self.lpf_range.text()).replace(os.linesep, ''))


class shrink_wrap(Feature):
    """
    This class encapsulates support feature.
    """
    def __init__(self):
        super(shrink_wrap, self).__init__()
        self.id = 'shrink wrap'


    def init_config(self, conf_map):
        """
        This function sets support feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'shrink_wrap_trigger' in conf_map:
            triggers = conf_map['shrink_wrap_trigger']
            self.active.setChecked(True)
            self.shrink_wrap_triggers.setText(str(triggers).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'shrink_wrap_type' in conf_map:
            self.shrink_wrap_type.setText(str(conf_map['shrink_wrap_type']).replace(" ", ""))
        else:
            self.shrink_wrap_type.setText('')
        if 'shrink_wrap_threshold' in conf_map:
            self.shrink_wrap_threshold.setText(str(conf_map['shrink_wrap_threshold']).replace(" ", ""))
        else:
            self.shrink_wrap_threshold.setText('')
        if 'shrink_wrap_gauss_sigma' in conf_map:
            self.shrink_wrap_gauss_sigma.setText(str(conf_map['shrink_wrap_gauss_sigma']).replace(" ", ""))
        else:
            self.shrink_wrap_gauss_sigma.setText('')


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.shrink_wrap_triggers = QLineEdit()
        layout.addRow("shrink wrap triggers", self.shrink_wrap_triggers)
        self.shrink_wrap_type = QLineEdit()
        layout.addRow("shrink wrap algorithm", self.shrink_wrap_type)
        self.shrink_wrap_threshold = QLineEdit()
        layout.addRow("shrink wrap threshold", self.shrink_wrap_threshold)
        self.shrink_wrap_gauss_sigma = QLineEdit()
        layout.addRow("shrink wrap Gauss sigma", self.shrink_wrap_gauss_sigma)


    def rec_default(self):
        """
        This function sets support feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.shrink_wrap_triggers.setText('[1,1]')
        self.shrink_wrap_type.setText('GAUSS')
        self.shrink_wrap_gauss_sigma.setText('1.0')
        self.shrink_wrap_threshold.setText('0.1')


    def add_feat_conf(self, conf_map):
        """
        This function adds support feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.shrink_wrap_triggers.text()) > 0:
            conf_map['shrink_wrap_trigger'] = ast.literal_eval(str(self.shrink_wrap_triggers.text()).replace(os.linesep,''))
        if len(self.shrink_wrap_type.text()) > 0:
            sw_type = str(self.shrink_wrap_type.text()).replace(' ','')
            # in case of multiple shrink wraps the shrink_wrap_type is a list of strings
            if sw_type.startswith('['):
                if sw_type.startswith('["') or sw_type.startswith(("['")):
                    conf_map['shrink_wrap_type'] = ast.literal_eval(sw_type)
                else: # parse as one string
                    sw_type = sw_type.replace('[', '["').replace(',', '","').replace(']', '"]')
                    conf_map['shrink_wrap_type'] = ast.literal_eval(sw_type)
            else:
                conf_map['shrink_wrap_type'] = sw_type
        if len(self.shrink_wrap_threshold.text()) > 0:
            conf_map['shrink_wrap_threshold'] = ast.literal_eval(str(self.shrink_wrap_threshold.text()))
        if len(self.shrink_wrap_gauss_sigma.text()) > 0:
            conf_map['shrink_wrap_gauss_sigma'] = ast.literal_eval(str(self.shrink_wrap_gauss_sigma.text()))


class phase_constrain(Feature):
    """
    This class encapsulates phase constrain feature.
    """
    def __init__(self):
        super(phase_constrain, self).__init__()
        self.id = 'phase constrain'


    def init_config(self, conf_map):
        """
        This function sets phase constrain feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'phc_trigger' in conf_map:
            triggers = conf_map['phc_trigger']
            self.active.setChecked(True)
            self.phase_triggers.setText(str(triggers).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'phc_phase_min' in conf_map:
            self.phc_phase_min.setText(str(conf_map['phc_phase_min']).replace(" ", ""))
        else:
            self.phc_phase_min.setText('')
        if 'phc_phase_max' in conf_map:
            self.phc_phase_max.setText(str(conf_map['phc_phase_max']).replace(" ", ""))
        else:
            self.phc_phase_max.setText('')


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.phase_triggers = QLineEdit()
        layout.addRow("phase constrain triggers", self.phase_triggers)
        self.phase_triggers.setToolTip('suggested trigger: [0, 1, <half iteration number>]')
        self.phc_phase_min = QLineEdit()
        layout.addRow("phase minimum", self.phc_phase_min)
        self.phc_phase_max = QLineEdit()
        layout.addRow("phase maximum", self.phc_phase_max)


    def rec_default(self):
        """
        This function sets phase constrain feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.phase_triggers.setText('[1,5,320]')
        self.phc_phase_min.setText('-1.57')
        self.phc_phase_max.setText('1.57')


    def add_feat_conf(self, conf_map):
        """
        This function adds phase constrain feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.phase_triggers.text()) > 0:
            conf_map['phc_trigger'] = ast.literal_eval(str(self.phase_triggers.text()).replace(os.linesep,''))
        if len(self.phc_phase_min.text()) > 0:
            conf_map['phc_phase_min'] = ast.literal_eval(str(self.phc_phase_min.text()))
        if len(self.phc_phase_max.text()) > 0:
            conf_map['phc_phase_max'] = ast.literal_eval(str(self.phc_phase_max.text()))


class pcdi(Feature):
    """
    This class encapsulates pcdi feature.
    """
    def __init__(self):
        super(pcdi, self).__init__()
        self.id = 'pcdi'


    def init_config(self, conf_map):
        """
        This function sets pcdi feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'pc_interval' in conf_map:
            self.active.setChecked(True)
            self.pc_interval.setText(str(conf_map['pc_interval']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'pc_type' in conf_map:
            self.pc_type.setText(str(conf_map['pc_type']).replace(" ", ""))
        else:
            self.pc_type.setText('')
        if 'pc_LUCY_iterations' in conf_map:
            self.pc_iter.setText(str(conf_map['pc_LUCY_iterations']).replace(" ", ""))
        else:
            self.pc_iter.setText('')
        if 'pc_normalize' in conf_map:
            self.pc_normalize.setText(str(conf_map['pc_normalize']).replace(" ", ""))
        else:
            self.pc_normalize.setText('')
        if 'pc_LUCY_kernel' in conf_map:
            self.pc_LUCY_kernel.setText(str(conf_map['pc_LUCY_kernel']).replace(" ", ""))
        else:
            self.pc_LUCY_kernel.setText('')


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.pc_interval = QLineEdit()
        layout.addRow("pc interval", self.pc_interval)
        self.pc_type = QLineEdit()
        layout.addRow("partial coherence algorithm", self.pc_type)
        self.pc_iter = QLineEdit()
        layout.addRow("LUCY iteration number", self.pc_iter)
        self.pc_normalize = QLineEdit()
        layout.addRow("normalize", self.pc_normalize)
        self.pc_LUCY_kernel = QLineEdit()
        layout.addRow("LUCY kernel area", self.pc_LUCY_kernel)


    def rec_default(self):
        """
        This function sets pcdi feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.pc_interval.setText('50')
        self.pc_type.setText('LUCY')
        self.pc_iter.setText('20')
        self.pc_normalize.setText('True')
        self.pc_LUCY_kernel.setText('[16, 16, 16]')


    def add_feat_conf(self, conf_map):
        """
        This function adds pcdi feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.pc_interval.text()) > 0:
            conf_map['pc_interval'] = ast.literal_eval(str(self.pc_interval.text()))
        if len(self.pc_type.text()) > 0:
            conf_map['pc_type'] = str(self.pc_type.text())
        if len(self.pc_iter.text()) > 0:
            conf_map['pc_LUCY_iterations'] = ast.literal_eval(str(self.pc_iter.text()))
        pc_normalize_txt = str(self.pc_normalize.text()).strip()
        if pc_normalize_txt == 'False':
            conf_map['pc_normalize'] = False
        else:
            conf_map['pc_normalize'] = True
        if len(self.pc_LUCY_kernel.text()) > 0:
            conf_map['pc_LUCY_kernel'] = ast.literal_eval(str(self.pc_LUCY_kernel.text()).replace(os.linesep,''))


class twin(Feature):
    """
    This class encapsulates twin feature.
    """
    def __init__(self):
        super(twin, self).__init__()
        self.id = 'twin'


    def init_config(self, conf_map):
        """
        This function sets twin feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'twin_trigger' in conf_map:
            self.active.setChecked(True)
            self.twin_triggers.setText(str(conf_map['twin_trigger']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return
        if 'twin_halves' in conf_map:
            self.twin_halves.setText(str(conf_map['twin_halves']).replace(" ", ""))
        else:
            self.twin_halves.setText('')


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.twin_triggers = QLineEdit()
        layout.addRow("twin triggers", self.twin_triggers)
        self.twin_halves = QLineEdit()
        layout.addRow("twin halves", self.twin_halves)


    def rec_default(self):
        """
        This function sets twin feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.twin_triggers.setText('[2]')
        self.twin_halves.setText('[0,0]')


    def add_feat_conf(self, conf_map):
        """
        This function adds twin feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if len(self.twin_triggers.text()) > 0:
            conf_map['twin_trigger'] = ast.literal_eval(str(self.twin_triggers.text()).replace(os.linesep,''))
        if len(self.twin_halves.text()) > 0:
            conf_map['twin_halves'] = ast.literal_eval(str(self.twin_halves.text()).replace(os.linesep,''))


class average(Feature):
    """
    This class encapsulates average feature.
    """
    def __init__(self):
        super(average, self).__init__()
        self.id = 'average'


    def init_config(self, conf_map):
        """
        This function sets average feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'average_trigger' in conf_map:
            self.active.setChecked(True)
            self.average_triggers.setText(str(conf_map['average_trigger']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.average_triggers = QLineEdit()
        layout.addRow("average triggers", self.average_triggers)


    def rec_default(self):
        """
        This function sets average feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.average_triggers.setText('[-50,1]')


    def add_feat_conf(self, conf_map):
        """
        This function adds average feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        conf_map['average_trigger'] = ast.literal_eval(str(self.average_triggers.text()).replace(os.linesep,''))


class progress(Feature):
    """
    This class encapsulates progress feature.
    """
    def __init__(self):
        super(progress, self).__init__()
        self.id = 'progress'


    def init_config(self, conf_map):
        """
        This function sets progress feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'progress_trigger' in conf_map:
            self.active.setChecked(True)
            self.progress_triggers.setText(str(conf_map['progress_trigger']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.progress_triggers = QLineEdit()
        layout.addRow("progress triggers", self.progress_triggers)


    def rec_default(self):
        """
        This function sets progress feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.progress_triggers.setText('[0,20]')


    def add_feat_conf(self, conf_map):
        """
        This function adds progress feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        conf_map['progress_trigger'] = ast.literal_eval(str(self.progress_triggers.text()).replace(os.linesep,''))


class live(Feature):
    """
    This class encapsulates live feature.
    """
    def __init__(self):
        super(live, self).__init__()
        self.id = 'live'


    def init_config(self, conf_map):
        """
        This function sets live feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        if 'live_trigger' in conf_map:
            self.active.setChecked(True)
            self.live_triggers.setText(str(conf_map['live_trigger']).replace(" ", ""))
        else:
            self.active.setChecked(False)
            return


    def fill_active(self, layout):
        """
        This function displays the feature's parameters when the feature becomes active.
        Parameters
        ----------
        layout : Layout widget
            a layout with the feature
        Returns
        -------
        nothing
        """
        self.live_triggers = QLineEdit()
        layout.addRow("live triggers", self.live_triggers)


    def rec_default(self):
        """
        This function sets live feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.live_triggers.setText('[0,5]')


    def add_feat_conf(self, conf_map):
        """
        This function adds live feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        conf_map['live_trigger'] = ast.literal_eval(str(self.live_triggers.text()).replace(os.linesep,''))


class Features(QWidget):
    """
    This class is composition of all feature classes.
    """
    def __init__(self, tab, layout):
        """
        Constructor, creates all concrete feature objects, and displays in window.
        """
        super(Features, self).__init__()
        feature_ids = ['GA', 'low resolution', 'shrink wrap', 'phase constrain', 'pcdi', 'twin', 'average', 'progress', 'live']
        self.leftlist = QListWidget()
        self.feature_dir = {'GA' : GA(),
                            'low resolution' : low_resolution(),
                            'shrink wrap' : shrink_wrap(),
                            'phase constrain' : phase_constrain(),
                            'pcdi' : pcdi(),
                            'twin' : twin(),
                            'average' : average(),
                            'progress' : progress(),
                            'live' : live()
                            }
        self.Stack = QStackedWidget(self)
        for i in range(len(feature_ids)):
            id = feature_ids[i]
            self.leftlist.insertItem(i, id)
            feature = self.feature_dir[id]
            feature.stackUI(self.leftlist.item(i), self)
            self.Stack.addWidget(feature.stack)

        layout.addWidget(self.leftlist)
        layout.addWidget(self.Stack)

        self.leftlist.currentRowChanged.connect(self.display)


    def display(self, i):
        self.Stack.setCurrentIndex(i)


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

        layout = QFormLayout()
        self.result_dir_button = QPushButton()
        layout.addRow("phasing results directory", self.result_dir_button)
        self.make_twin = QCheckBox('make twin')
        self.make_twin.setChecked(False)
        layout.addWidget(self.make_twin)
        self.unwrap = QCheckBox('include unwrapped phase')
        self.unwrap.setChecked(False)
        layout.addWidget(self.unwrap)
        self.imcrop = QComboBox()
        self.imcrop.addItem("none")
        self.imcrop.addItem("tight")
        self.imcrop.addItem("fraction")
        layout.addRow("image crop", self.imcrop)
        sub_layout = QFormLayout()
        self.set_imcrop_layout(sub_layout)
        layout.addRow(sub_layout)
        self.rampups = QLineEdit()
        layout.addRow("ramp upscale", self.rampups)
        self.complex_mode = QComboBox()
        self.complex_mode.addItem("AmpPhase")
        self.complex_mode.addItem("ReIm")
        layout.addRow("complex mode", self.complex_mode)
        self.interpolation_mode = QComboBox()
        self.interpolation_mode.addItem("no interpolation")
        self.interpolation_mode.addItem("AmpPhase")
        self.interpolation_mode.addItem("ReIm")
        layout.addRow("interpolation mode", self.interpolation_mode)
        self.interpolation_resolution = QLineEdit()
        layout.addRow("interpolation resolution", self.interpolation_resolution)
        self.interpolation_resolution.setToolTip('Supported values: "min_deconv_res", int value, float value, list')
        self.determine_resolution = QLineEdit()
        layout.addRow("determine resolution", self.determine_resolution)
        self.determine_resolution.setToolTip('If present, the resolution in direct and reciprocal spaces will be found. Supported value: "deconv"')
        self.resolution_deconv_contrast = QLineEdit()
        layout.addRow("determine resolution", self.resolution_deconv_contrast)
        self.resolution_deconv_contrast.setToolTip('float number < 1.0')
        self.write_recip = QCheckBox('write reciprocal')
        self.write_recip.setChecked(False)
        layout.addWidget(self.write_recip)
        cmd_layout = QHBoxLayout()
        self.set_disp_conf_from_button = QPushButton("Load disp conf from")
        self.set_disp_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        self.config_disp = QPushButton('process display', self)
        self.config_disp.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_disp_conf_from_button)
        cmd_layout.addWidget(self.config_disp)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.imcrop.currentIndexChanged.connect(lambda: self.set_imcrop_layout(sub_layout))
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

        if 'imcrop' not in conf_map:
            self.imcrop.setCurrentIndex(0)
        elif conf_map['imcrop'] == 'tight':
            self.imcrop.setCurrentIndex(1)
            if 'imcrop_margin' in conf_map:
                self.imcrop_margin.setText(str(conf_map['imcrop_margin']).replace(" ", ""))
            if 'imcrop_thresh' in conf_map:
                self.imcrop_thresh.setText(str(conf_map['imcrop_thresh']).replace(" ", ""))
        elif conf_map['imcrop'] == 'fraction':
            self.imcrop.setCurrentIndex(2)
            if 'imcrop_fraction' in conf_map:
                self.imcrop_fraction.setText(str(conf_map['imcrop_fraction']).replace(" ", ""))

        if 'rampups' in conf_map:
            self.rampups.setText(str(conf_map['rampups']).replace(" ", ""))

        if 'complex_mode' in conf_map and conf_map['complex_mode'] == 'ReIm':
            self.complex_mode.setCurrentIndex(1)
        else:
            self.complex_mode.setCurrentIndex(0)

        if 'interpolation_mode' in conf_map:
            if conf_map['interpolation_mode'] == 'ReIm':
                self.interpolation_mode.setCurrentIndex(1)
            elif conf_map['interpolation_mode'] == 'AmpPhase':
                self.interpolation_mode.setCurrentIndex(1)
            else:
                self.interpolation_mode.setCurrentIndex(0)
        else:
            self.interpolation_mode.setCurrentIndex(0)

        if 'interpolation_resolution' in conf_map:
            self.interpolation_resolution.setText(str(conf_map['interpolation_resolution']).replace(" ", ""))

        if 'determine_resolution' in conf_map:
            self.determine_resolution.setText(str(conf_map['determine_resolution']).replace(" ", ""))
        if 'resolution_deconv_contrast' in conf_map:
            self.resolution_deconv_contrast.setText(str(conf_map['resolution_deconv_contrast']).replace(" ", ""))

        if 'write_recip' in conf_map:
            write_recip = conf_map['write_recip']
            if write_recip:
                self.write_recip.setChecked(True)
            else:
                self.write_recip.setChecked(False)
        else:
            self.write_recip.setChecked(False)


    def clear_conf(self):
        self.result_dir_button.setText('')
        self.make_twin.setChecked(False)
        self.unwrap.setChecked(False)
        self.imcrop.setCurrentIndex(0)
        self.rampups.setText('')
        self.complex_mode.setCurrentIndex(0)
        self.interpolation_mode.setCurrentIndex(0)
        self.interpolation_resolution.setText('')
        self.determine_resolution.setText('')
        self.resolution_deconv_contrast.setText('')
        self.write_recip.setChecked(False)


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
            msg_window('please select valid disp config file')


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

        if self.imcrop.currentIndex() == 1:
            conf_map['imcrop'] = 'tight'
            if len(self.imcrop_margin.text()) > 0:
                conf_map['imcrop_margin'] = ast.literal_eval(str(self.imcrop_margin.text()))
            if len(self.imcrop_thresh.text()) > 0:
                conf_map['imcrop_thresh'] = ast.literal_eval(str(self.imcrop_thresh.text()))
        if self.imcrop.currentIndex() == 2:
            conf_map['imcrop'] = 'fraction'
            if len(self.imcrop_fraction.text()) > 0:
                conf_map['imcrop_fraction'] = ast.literal_eval(str(self.imcrop_fraction.text()))

        if len(self.rampups.text()) > 0:
            conf_map['rampups'] = ast.literal_eval(str(self.rampups.text()).replace(os.linesep, ''))

        if self.complex_mode.currentIndex() == 0:
            conf_map['complex_mode'] = 'AmpPhase'
        if self.complex_mode.currentIndex() == 1:
            conf_map['complex_mode'] = 'ReIm'

        if self.interpolation_mode.currentIndex() == 1:
            conf_map['interpolation_mode'] = 'AmpPhase'
        if self.interpolation_mode.currentIndex() == 2:
            conf_map['interpolation_mode'] = 'ReIm'

        if len(self.interpolation_resolution.text()) > 0:
            try:
                conf_map['interpolation_resolution'] = ast.literal_eval(str(self.interpolation_resolution.text()))
            except ValueError:
                conf_map['interpolation_resolution'] = str(self.interpolation_resolution.text())

        if len(self.determine_resolution.text()) > 0:
            conf_map['determine_resolution'] = str(self.determine_resolution.text())
        if len(self.resolution_deconv_contrast.text()) > 0:
            conf_map['resolution_deconv_contrast'] = ast.literal_eval(str(self.resolution_deconv_contrast.text()))

        if self.write_recip.isChecked():
            conf_map['write_recip'] = True

        return conf_map


    def set_imcrop_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        if self.imcrop.currentIndex() == 1:
            self.imcrop_margin = QLineEdit()
            layout.addRow("margin added to extend", self.imcrop_margin)
            self.imcrop_thresh = QLineEdit()
            layout.addRow("threshold extend", self.imcrop_thresh)
        elif self.imcrop.currentIndex() == 2:
            self.imcrop_fraction = QLineEdit()
            layout.addRow("image crop fractions", self.imcrop_fraction)


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

        conf_map = self.get_disp_config()
        er_msg = ut.verify('config_disp', conf_map)
        if len(er_msg) > 0:
            msg_window(er_msg)
            if not self.main_win.no_verify:
                return
        if len(conf_map) > 0:
            ut.write_config(conf_map, ut.join(self.main_win.experiment_dir, 'conf', 'config_disp'))

        self.tabs.run_viz()


    def save_conf(self):
        if not self.main_win.is_exp_exists():
            msg_window('the experiment does not exist, cannot save the config_disp file')
            return

        conf_map = self.get_disp_config()
        er_msg = ut.verify('config_disp', conf_map)
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
        results_dir = self.main_win.experiment_dir
        if not self.main_win.separate_scans.isChecked() and \
                not self.main_win.separate_scan_ranges.isChecked():
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
            self.result_dir_button.setText('')
            msg_window('please select valid results directory')


class MpTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(MpTab, self).__init__(parent)
        self.name = 'Multi peak'
        self.conf_name = 'config_mp'


    def init(self, tabs, main_window):
        """
        Creates and initializes the 'data' tab.
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
        self.switch_peak_trigger = QLineEdit()
        layout.addRow("switch peak trigger", self.switch_peak_trigger)
        self.adapt_trigger = QLineEdit()
        layout.addRow("adapt trigger", self.adapt_trigger)
        self.scan = QLineEdit()
        layout.addRow("scan(s)", self.scan)
        self.orientations = QLineEdit()
        layout.addRow("peak orientations", self.orientations)
        self.hkl_in = QLineEdit()
        layout.addRow("hkl in", self.hkl_in)
        self.hkl_out = QLineEdit()
        layout.addRow("hkl out", self.hkl_out)
        self.twin_plane = QLineEdit()
        layout.addRow("twin plane", self.twin_plane)
        self.sample_axis = QLineEdit()
        layout.addRow("sample axis", self.sample_axis)
        self.lattice_size = QLineEdit()
        layout.addRow("lattice size", self.lattice_size)
        self.final_size = QLineEdit()
        layout.addRow("final size", self.final_size)
        self.adapt_threshold_init = QLineEdit()
        layout.addRow("adapt threshold init", self.adapt_threshold_init)
        self.adapt_threshold_iters = QLineEdit()
        layout.addRow("adapt threshold iters", self.adapt_threshold_iters)
        self.adapt_threshold_vals = QLineEdit()
        layout.addRow("adapt threshold vals", self.adapt_threshold_vals)
        self.weight_init = QLineEdit()
        layout.addRow("initial weight", self.weight_init)
        self.weight_iters = QLineEdit()
        layout.addRow("weight iters", self.weight_iters)
        self.weight_vals = QLineEdit()
        layout.addRow("weight vals", self.weight_vals)
        self.adapt_alien_start = QLineEdit()
        layout.addRow("adapt alien start", self.adapt_alien_start)
        self.adapt_alien_threshold = QLineEdit()
        layout.addRow("adapt alien threshold", self.adapt_alien_threshold)
        self.adapt_power = QLineEdit()
        layout.addRow("adapt power", self.adapt_power)

        cmd_layout = QHBoxLayout()
        self.set_mp_conf_from_button = QPushButton("Load conf from")
        self.set_mp_conf_from_button.setStyleSheet("background-color:rgb(205,178,102)")
        cmd_layout.addWidget(self.set_mp_conf_from_button)
        self.set_params_button = QPushButton("Save parameters")
        self.set_params_button.setStyleSheet("background-color:rgb(175,208,156)")
        cmd_layout.addWidget(self.set_params_button)
        layout.addRow(cmd_layout)
        self.setLayout(layout)

        self.set_mp_conf_from_button.clicked.connect(self.load_mp_conf)
        self.set_params_button.clicked.connect(self.save_conf)


    def run_tab(self):
        pass


    def clear_conf(self):
        self.switch_peak_trigger.setText('')
        self.adapt_trigger.setText('')
        self.scan.setText('')
        self.orientations.setText('')
        self.hkl_in.setText('')
        self.hkl_out.setText('')
        self.twin_plane.setText('')
        self.sample_axis.setText('')
        self.lattice_size.setText('')
        self.final_size.setText('')
        self.adapt_threshold_init.setText('')
        self.adapt_threshold_iters.setText('')
        self.adapt_threshold_vals.setText('')
        self.weight_init.setText('')
        self.weight_iters.setText('')
        self.weight_vals.setText('')
        self.adapt_alien_start.setText('')
        self.adapt_alien_threshold.setText('')
        self.adapt_power.setText('')


    def load_tab(self, conf_map):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : dict
            configuration (config_data)
        Returns
        -------
        nothing
        """
        if 'switch_peak_trigger' in conf_map:
            self.switch_peak_trigger.setText(str(conf_map['switch_peak_trigger']))
        if 'adapt_trigger' in conf_map:
            self.adapt_trigger.setText(str(conf_map['adapt_trigger']))
        if 'scan' in conf_map:
            self.scan.setText(str(conf_map['scan']).replace(" ", ""))
        if 'orientations' in conf_map:
            self.orientations.setText(str(conf_map['orientations']))
        if 'hkl_in' in conf_map:
            self.hkl_in.setText(str(conf_map['hkl_in']))
        if 'hkl_out' in conf_map:
            self.hkl_out.setText(str(conf_map['hkl_out']))
        if 'twin_plane' in conf_map:
            self.twin_plane.setText(str(conf_map['twin_plane']))
        if 'sample_axis' in conf_map:
            self.sample_axis.setText(str(conf_map['sample_axis']))
        if 'lattice_size' in conf_map:
            self.lattice_size.setText(str(conf_map['lattice_size']))
        if 'final_size' in conf_map:
            self.final_size.setText(str(conf_map['final_size']))
        if 'adapt_threshold_init' in conf_map:
            self.adapt_threshold_init.setText(str(conf_map['adapt_threshold_init']))
        if 'adapt_threshold_iters' in conf_map:
            self.adapt_threshold_iters.setText(str(conf_map['adapt_threshold_iters']))
        if 'adapt_threshold_vals' in conf_map:
            self.adapt_threshold_vals.setText(str(conf_map['adapt_threshold_vals']))
        if 'weight_init' in conf_map:
            self.weight_init.setText(str(conf_map['weight_init']))
        if 'weight_iters' in conf_map:
            self.weight_iters.setText(str(conf_map['weight_iters']))
        if 'weight_vals' in conf_map:
            self.weight_vals.setText(str(conf_map['weight_vals']))
        if 'adapt_alien_start' in conf_map:
            self.adapt_alien_start.setText(str(conf_map['adapt_alien_start']))
        if 'adapt_alien_threshold' in conf_map:
            self.adapt_alien_threshold.setText(str(conf_map['adapt_alien_threshold']))
        if 'adapt_power' in conf_map:
            self.adapt_power.setText(str(conf_map['adapt_power']))


    def save_conf(self):
        """
        It reads parameters related to multi peak from the window and saves in config_mp file.
        Parameters
        ----------
        none
        Returns
        -------
        conf_map : dict
            contains parameters read from window
        """
        conf_map = {}

        if len(self.switch_peak_trigger.text()) > 0:
            conf_map['switch_peak_trigger'] = ast.literal_eval(str(self.switch_peak_trigger.text()))
        if len(self.adapt_trigger.text()) > 0:
            conf_map['adapt_trigger'] = ast.literal_eval(str(self.adapt_trigger.text()))
        if len(self.scan.text()) > 0:
            conf_map['scan'] = str(self.scan.text())
        if len(self.orientations.text()) > 0:
            conf_map['orientations'] = ast.literal_eval(str(self.orientations.text()))
        if len(self.hkl_in.text()) > 0:
            conf_map['hkl_in'] = ast.literal_eval(str(self.hkl_in.text()))
        if len(self.hkl_out.text()) > 0:
            conf_map['hkl_out'] = ast.literal_eval(str(self.hkl_out.text()))
        if len(self.twin_plane.text()) > 0:
            conf_map['twin_plane'] = ast.literal_eval(str(self.twin_plane.text()))
        if len(self.sample_axis.text()) > 0:
            conf_map['sample_axis'] = ast.literal_eval(str(self.sample_axis.text()))
        if len(self.lattice_size.text()) > 0:
            conf_map['lattice_size'] = ast.literal_eval(str(self.lattice_size.text()))
        if len(self.final_size.text()) > 0:
            conf_map['final_size'] = ast.literal_eval(str(self.final_size.text()))
        if len(self.adapt_threshold_init.text()) > 0:
            conf_map['adapt_threshold_init'] = ast.literal_eval(str(self.adapt_threshold_init.text()))
        if len(self.adapt_threshold_iters.text()) > 0:
            conf_map['adapt_threshold_iters'] = ast.literal_eval(str(self.adapt_threshold_iters.text()))
        if len(self.adapt_threshold_vals.text()) > 0:
            conf_map['adapt_threshold_vals'] = ast.literal_eval(str(self.adapt_threshold_vals.text()))
        if len(self.adapt_threshold_init.text()) > 0:
            conf_map['weight_init'] = ast.literal_eval(str(self.weight_init.text()))
        if len(self.weight_iters.text()) > 0:
            conf_map['weight_iters'] = ast.literal_eval(str(self.weight_iters.text()))
        if len(self.weight_vals.text()) > 0:
            conf_map['weight_vals'] = ast.literal_eval(str(self.weight_vals.text()))
        if len(self.adapt_alien_start.text()) > 0:
            conf_map['adapt_alien_start'] = ast.literal_eval(str(self.adapt_alien_start.text()))
        if len(self.adapt_alien_threshold.text()) > 0:
            conf_map['adapt_alien_threshold'] = ast.literal_eval(str(self.adapt_alien_threshold.text()))
        if len(self.adapt_power.text()) > 0:
            conf_map['adapt_power'] = ast.literal_eval(str(self.adapt_power.text()))

        ut.write_config(conf_map, self.main_win.experiment_dir + '/conf/config_mp')


    def load_mp_conf(self):
        """
        It displays a select dialog for user to select a configuration file. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        conf_file = select_file(os.getcwd())
        if conf_file is not None:
            conf_map = ut.read_config(conf_file.replace(os.sep, '/'))
            self.load_tab(conf_map)
        else:
            msg_window('please select valid config file')


def main():
    """
    Starts GUI application.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_verify", action="store_true",
                        help="if True the verifier has no effect on processing")
    parser.add_argument("--debug", action="store_true",
                        help="if True the exceptions are not handled")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    ex = cdi_gui()
    ex.set_args(sys.argv[1:], no_verify=args.no_verify, debug=args.debug)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
