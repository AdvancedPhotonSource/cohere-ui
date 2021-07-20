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
import shutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cohere.src_py.utilities.utils as ut
import importlib
import config_verifier as ver


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
    dialog = QFileDialog(None, 'select dir', start_dir)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setSidebarUrls([QUrl.fromLocalFile(start_dir)])
    if dialog.exec_() == QDialog.Accepted:
        return str(dialog.selectedFiles()[0])
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
    dialog = QFileDialog(None, 'select dir', start_dir)
    dialog.setFileMode(QFileDialog.DirectoryOnly)
    dialog.setSidebarUrls([QUrl.fromLocalFile(start_dir)])
    if dialog.exec_() == QDialog.Accepted:
        return str(dialog.selectedFiles()[0])
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


def write_conf(conf_map, dir, file):
    """
    It creates configuration file from the parameters included in dictionary, verifies, and saves in the configuration directory.
    Parameters
    ----------
    conf_map : dict
        dictionary containing configuration parameters
    dir : str
        a directory where the configuration file will be saved
    file : str
        name of the configuration file to save
    Returns
    -------
    nothing
    """
    # create "temp" file first, verify it, and if ok, copy to a configuration file
    if not os.path.exists(dir):
        os.makedirs(dir)
    conf_file = os.path.join(dir, file)
    temp_file = os.path.join(dir, 'temp')
    if os.path.isfile(temp_file):
        os.remove(temp_file)
    with open(temp_file, 'a') as f:
        for key in conf_map:
            value = conf_map[key]
            if len(value) > 0:
                f.write(key + ' = ' + conf_map[key] + '\n')
    f.close()

    if file == 'config':
        if not ver.ver_config(temp_file):
            os.remove(temp_file)
            msg_window('please check the entries in the main window. Cannot save this format')
            return False
    elif file == 'config_prep':
        if not ver.ver_config_prep(temp_file):
            os.remove(temp_file)
            msg_window('please check the entries in the Data prep tab. Cannot save this format')
            return False
    elif file == 'config_data':
        if not ver.ver_config_data(temp_file):
            os.remove(temp_file)
            msg_window('please check the entries in the Data tab. Cannot save this format')
            return False
    elif file.endswith('config_rec'):
        if not ver.ver_config_rec(temp_file):
            os.remove(temp_file)
            msg_window('please check the entries in the Reconstruction tab. Cannot save this format')
            return False
    elif file == 'config_disp':
        if not ver.ver_config_disp(temp_file):
            os.remove(temp_file)
            msg_window('please check the entries in the Display tab. Cannot save this format')
            return False
    # copy if verified
    shutil.copy(temp_file, conf_file)
    os.remove(temp_file)
    return True


class cdi_gui(QWidget):
    def __init__(self, parent=None):
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

        self.beamline = None
        self.exp_id = None
        self.experiment_dir = None
        self.working_dir = None
        self.specfile = None

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
        self.spec_file_button = QPushButton()
        ruplayout.addRow("spec file", self.spec_file_button)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(uplayout)

        self.t = None
        # self.vbox.addWidget(self.t)

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
        self.setWindowTitle("CDI Reconstruction")

        self.set_exp_button.clicked.connect(self.load_experiment)
        self.set_work_dir_button.clicked.connect(self.set_working_dir)
        self.spec_file_button.clicked.connect(self.set_spec_file)
        self.run_button.clicked.connect(self.run_everything)
        self.create_exp_button.clicked.connect(self.set_experiment)


    def set_args(self, args):
        self.args = args


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
        self.specfile = select_file(os.getcwd())
        if self.specfile is not None:
            self.spec_file_button.setStyleSheet("Text-align:left")
            self.spec_file_button.setText(self.specfile)
        else:
            self.specfile = None
            self.spec_file_button.setText('')
        if self.is_exp_exists() or self.is_exp_set():
            self.set_experiment()


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
            exp_id = exp_id + '_' + scan
        if not os.path.exists(os.path.join(self.working_dir, exp_id)):
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
        load_dir = select_dir(os.getcwd())
        if load_dir is not None:
            if os.path.isfile(os.path.join(load_dir, 'conf', 'config')):
                self.load_main(load_dir)
            else:
                msg_window('missing conf/config file, not experiment directory')
                return

            if self.t is None:
                self.t = Tabs(self)
                self.vbox.addWidget(self.t)
            self.t.clear_configs()
            self.t.load_conf(load_dir)

            self.set_experiment(True)
        else:
            msg_window('please select valid conf directory')


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
            return


    def load_main(self, load_dir):
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
        conf = os.path.join(load_dir, 'conf', 'config')
        try:
            conf_map = ut.read_config(conf)
        except Exception:
            msg_window('please check configuration file ' + conf + '. Cannot parse, ')
            return

        try:
            self.scan = conf_map.scan
            self.scan_widget.setText(self.scan)
        except:
            self.scan = None
        try:
            self.id = conf_map.experiment_id
            self.Id_widget.setText(self.id)
            if self.scan != None:
                self.exp_id = self.id + '_' + self.scan
            else:
                self.exp_id = self.id
        except:
            self.scan = None
            self.id = None
            self.exp_id = None
            msg_window('id is not configured in ' + conf + ' file.')

        try:
            working_dir = conf_map.working_dir
            if not os.path.isdir(working_dir):
                self.working_dir = None
                self.set_work_dir_button.setText('')
                msg_window('The working directory ' + working_dir + ' from config file does not exist. Select valid working directory and set experiment')
            else:
                self.working_dir = conf_map.working_dir
                self.set_work_dir_button.setStyleSheet("Text-align:left")
                self.set_work_dir_button.setText(self.working_dir)
        except:
            pass

        if self.working_dir is not None and self.exp_id is not None:
            self.experiment_dir = os.path.join(self.working_dir, self.exp_id)

        try:
            specfile = conf_map.specfile
            if os.path.isfile(specfile):
                self.specfile = conf_map.specfile
                self.spec_file_button.setStyleSheet("Text-align:left")
                self.spec_file_button.setText(self.specfile)
            else:
                msg_window('The specfile file ' + specfile + ' in config file does not exist')
        except:
            self.specfile = None
            self.spec_file_button.setText('')

        try:
            self.beamline = conf_map.beamline
            self.beamline_widget.setText(conf_map.beamline)
        except:
            pass


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
        experiment_conf_dir = os.path.join(self.experiment_dir, 'conf')
        if not os.path.exists(experiment_conf_dir):
            os.makedirs(experiment_conf_dir)


    def set_experiment(self, new_exp=False):
        """
        Reads the parameters in the window, and sets the experiment to this values, i.e. creates experiment directory,
        and saves all configuration files with parameters from window.

        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.id = str(self.Id_widget.text()).strip()
        if self.id == '' or self.working_dir is None:
            msg_window('id and working directory must be entered')
            return
        conf_map = {}

        self.scan = str(self.scan_widget.text()).replace(' ','')
        if len(self.scan) > 0:
            scans = self.scan.split('-')
            if len(scans) > 2:
                msg_window('if entering scan or scan range, please enter numeric values, separated with "-" if range')
                return
            for sc in scans:
                try:
                    numeric = int(sc)
                except:
                    msg_window('if entering scan or scan range, please enter numeric values, separated with "-" if range')
                    return
            conf_map['scan'] = '"' + self.scan + '"'
            self.exp_id = self.id + '_' + self.scan
            
        else:
            self.exp_id = self.id
        self.experiment_dir = os.path.join(self.working_dir, self.exp_id)
        self.assure_experiment_dir()

        if not new_exp:
            # read the configurations from GUI and write to experiment config files
            # save the main config
            conf_map['working_dir'] = '"' + str(self.working_dir).strip() + '"'
            conf_map['experiment_id'] = '"' + self.id + '"'
            if len(self.beamline_widget.text().strip()) > 0:
                conf_map['beamline'] = '"' + str(self.beamline_widget.text().strip()) + '"'
                self.beamline = self.beamline_widget.text().strip()
            if self.specfile is not None:
                conf_map['specfile'] = '"' + str(self.specfile).strip() + '"'
            write_conf(conf_map, os.path.join(self.experiment_dir, 'conf'), 'config')

        if self.t is None:
            try:
                self.t = Tabs(self)
                self.vbox.addWidget(self.t)
            except:
                pass

        if not new_exp:
            self.t.save_conf()


class Tabs(QTabWidget):
    """
    The main window contains four tabs, each tab holding parameters for different part of processing.
    The tabs are as follows: prep (prepare data), data (format data), rec (reconstruction), disp (visualization).
    This class holds holds the tabs.
    """
    def __init__(self, main_win, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(Tabs, self).__init__(parent)
        self.main_win = main_win

        if self.main_win.beamline is not None:
            try:
                beam = importlib.import_module('beamlines.' + self.main_win.beamline + '.beam_tabs')
            except Exception as e:
                print (e)
                msg_window('cannot import beamlines.' + self.main_win.beamline + ' module' )
                raise
            self.prep_tab = beam.PrepTab()
            self.format_tab = DataTab()
            self.rec_tab = RecTab()
            self.display_tab = beam.DispTab()
            self.tabs = [self.prep_tab, self.format_tab, self.rec_tab, self.display_tab]
        else:
            self.format_tab = DataTab()
            self.rec_tab = RecTab()
            self.tabs = [self.format_tab, self.rec_tab]

        for tab in self.tabs:
            self.addTab(tab, tab.name)
            tab.init(self, main_win)


    def notify(self, **args):
        try:
            self.display_tab.update_tab(**args)
        except:
            pass


    def clear_configs(self):
        for tab in self.tabs:
            tab.clear_conf()


    def run_all(self):
        for tab in self.tabs:
            tab.run_tab()

    def run_prep(self):
        import run_prep as prep

        # this line is passing all parameters from command line to prep script. 
        # if there are other parameters, one can add some code here
        prep.handle_prep(self.main_win.experiment_dir, self.main_win.args)

    def run_viz(self):
        import run_disp as dp

        dp.handle_visualization(self.main_win.experiment_dir)


    def load_conf(self, load_dir):
        for tab in self.tabs:
            tab.load_tab(load_dir)


    def save_conf(self):
        for tab in self.tabs:
            tab.save_conf()


class DataTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(DataTab, self).__init__(parent)
        self.name = 'Data'


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
        self.amp_intensity = QLineEdit()
        layout.addRow("amp intensity", self.amp_intensity)
        self.center_shift = QLineEdit()
        layout.addRow("center_shift", self.center_shift)
        self.adjust_dimensions = QLineEdit()
        layout.addRow("pad, crop", self.adjust_dimensions)
        self.binning = QLineEdit()
        layout.addRow("binning", self.binning)
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
        self.amp_intensity.setText('')
        self.binning.setText('')
        self.center_shift.setText('')
        self.adjust_dimensions.setText('')


    def load_tab(self, load_item):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : str
            configuration file (config_data)
        Returns
        -------
        nothing
        """
        if os.path.isfile(load_item):
            conf = load_item
        else:
            conf = os.path.join(load_item, 'conf', 'config_data')
            if not os.path.isfile(conf):
                msg_window('info: the load directory does not contain config_data file')
                return
#        if not ver.ver_config_data(conf):
#            msg_window('please check configuration file ' + conf + '. Cannot parse, ')
#            return
        try:
            conf_map = ut.read_config(conf)
        except Exception as e:
            msg_window('please check configuration file ' + conf + '. Cannot parse, ' + str(e))
            return
        alg = 'none'
        try:
            alg = str(conf_map.alien_alg)
        except AttributeError:
            self.alien_alg.setCurrentIndex(0)
        if alg == 'none':
            self.alien_alg.setCurrentIndex(0)
        elif alg == 'block_aliens':
            self.alien_alg.setCurrentIndex(1)
            try:
                self.aliens.setText(str(conf_map.aliens).replace(" ", ""))
            except AttributeError:
                pass
        elif alg == 'alien_file':
            self.alien_alg.setCurrentIndex(2)
            try:
                self.alien_file.setText(str(conf_map.alien_file).replace(" ", ""))
            except AttributeError:
                pass
        elif alg == 'AutoAlien1':
            self.alien_alg.setCurrentIndex(3)
            try:
                self.AA1_size_threshold.setText(str(conf_map.AA1_size_threshold).replace(" ", ""))
            except AttributeError:
                pass
            try:
                self.AA1_asym_threshold.setText(str(conf_map.AA1_asym_threshold).replace(" ", ""))
            except AttributeError:
                pass
            try:
                self.AA1_min_pts.setText(str(conf_map.AA1_min_pts).replace(" ", ""))
            except AttributeError:
                pass
            try:
                self.AA1_eps.setText(str(conf_map.AA1_eps).replace(" ", ""))
            except AttributeError:
                pass
            try:
                self.AA1_amp_threshold.setText(str(conf_map.AA1_amp_threshold).replace(" ", ""))
            except AttributeError:
                pass
            try:
                self.AA1_save_arrs.setChecked(conf_map.AA1_save_arrs)
            except AttributeError:
                self.AA1_save_arrs.setChecked(False)
            try:
                self.AA1_expandcleanedsigma.setText(str(conf_map.AA1_expandcleanedsigma).replace(" ", ""))
            except AttributeError:
                pass
        try:
            self.amp_intensity.setText(str(conf_map.amp_threshold).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.binning.setText(str(conf_map.binning).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.center_shift.setText(str(conf_map.center_shift).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.adjust_dimensions.setText(str(conf_map.adjust_dimensions).replace(" ", ""))
        except AttributeError:
            pass


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
            conf_map['alien_alg'] = '"block_aliens"'
            if len(self.aliens.text()) > 0:
                conf_map['aliens'] = str(self.aliens.text()).replace('\n', '')
        if self.alien_alg.currentIndex() == 2:
            conf_map['alien_alg'] = '"alien_file"'
            if len(self.alien_file.text()) > 0:
                conf_map['alien_file'] = '"' + str(self.alien_file.text()) + '"'
        elif self.alien_alg.currentIndex() == 3:
            conf_map['alien_alg'] = '"AutoAlien1"'
            if len(self.AA1_size_threshold.text()) > 0:
                conf_map['AA1_size_threshold'] = str(self.AA1_size_threshold.text())
            if len(self.AA1_asym_threshold.text()) > 0:
                conf_map['AA1_asym_threshold'] = str(self.AA1_asym_threshold.text())
            if len(self.AA1_min_pts.text()) > 0:
                conf_map['AA1_min_pts'] = str(self.AA1_min_pts.text())
            if len(self.AA1_eps.text()) > 0:
                conf_map['AA1_eps'] = str(self.AA1_eps.text())
            if len(self.AA1_amp_threshold.text()) > 0:
                conf_map['AA1_amp_threshold'] = str(self.AA1_amp_threshold.text())
            if self.AA1_save_arrs.isChecked():
                conf_map['AA1_save_arrs'] = "True"
            if len(self.AA1_expandcleanedsigma.text()) > 0:
                conf_map['AA1_expandcleanedsigma'] = str(self.AA1_expandcleanedsigma.text())

        if len(self.amp_intensity.text()) > 0:
            conf_map['amp_threshold'] = str(self.amp_intensity.text())
        if len(self.binning.text()) > 0:
            conf_map['binning'] = str(self.binning.text()).replace('\n', '')
        if len(self.center_shift.text()) > 0:
            conf_map['center_shift'] = str(self.center_shift.text()).replace('\n', '')
        if len(self.adjust_dimensions.text()) > 0:
            conf_map['adjust_dimensions'] = str(self.adjust_dimensions.text()).replace('\n', '')

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


    def set_alien_file(self):
        """
        It display a select dialog for user to select an alien file.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.alien_filename = select_file(self.alien_filename)
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
        import format_data as run_dt

        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
        elif not self.main_win.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
        elif len(self.amp_intensity.text()) == 0:
            msg_window('Please, enter amp intensity parameter')
        else:
            found_file = False
            for p, d, f in os.walk(self.main_win.experiment_dir):
                if 'prep_data.tif' in f:
                    found_file = True
                    break
            if found_file:
                conf_map = self.get_data_config()
                conf_dir = os.path.join(self.main_win.experiment_dir, 'conf')
                if write_conf(conf_map, conf_dir, 'config_data'):
                    run_dt.data(self.main_win.experiment_dir)
            else:
                msg_window('Please, run data preparation in previous tab to activate this function')


    def save_conf(self):
        # save data config
        conf_map = self.get_data_config()
        if len(conf_map) > 0:
            write_conf(conf_map, os.path.join(self.main_win.experiment_dir, 'conf'), 'config_data')


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
            self.load_tab(data_file)
        else:
            msg_window('please select valid data config file')


class RecTab(QWidget):
    def __init__(self, parent=None):
        """
        Constructor, initializes the tabs.
        """
        super(RecTab, self).__init__(parent)
        self.name = 'Reconstruction'


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
        self.old_conf_id = ''

        layout = QVBoxLayout()
        ulayout = QFormLayout()
        mlayout = QHBoxLayout()

        hbox = QHBoxLayout()
        self.cont = QCheckBox('continue')
        self.cont.setChecked(False)
        hbox.addWidget(self.cont)
        self.cont_dir_label = QLabel('    cont dir')
        hbox.addWidget(self.cont_dir_label)
        self.cont_dir_label.hide()
        self.cont_dir_button = QPushButton()
        hbox.addWidget(self.cont_dir_button)
        self.cont_dir_button.hide()
        ulayout.addRow(hbox)

        self.add_conf_button = QPushButton('add configuration', self)
        ulayout.addWidget(self.add_conf_button)
        self.rec_id = QComboBox()
        self.rec_id.InsertAtBottom
        self.rec_id.addItem("main")
        ulayout.addWidget(self.rec_id)
        self.rec_id.hide()
        self.proc = QComboBox()
        if sys.platform != 'darwin':
            self.proc.addItem("cuda")
        self.proc.addItem("opencl")
        self.proc.addItem("cpu")
        ulayout.addRow("processor type", self.proc)
        self.device = QLineEdit()
        ulayout.addRow("device(s)", self.device)
        self.reconstructions = QLineEdit()
        ulayout.addRow("number of reconstructions", self.reconstructions)
        self.alg_seq = QLineEdit()
        ulayout.addRow("algorithm sequence", self.alg_seq)
        # TODO add logic to show this only if HIO is in sequence
        self.beta = QLineEdit()
        ulayout.addRow("beta", self.beta)
        self.support_area = QLineEdit()
        ulayout.addRow("support_area", self.support_area)
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

        self.cont_dir_button.clicked.connect(self.set_cont_dir)
        self.config_rec_button.clicked.connect(self.run_tab)
        self.cont.stateChanged.connect(self.toggle_cont)
        self.rec_default_button.clicked.connect(self.set_defaults)
        self.add_conf_button.clicked.connect(self.add_rec_conf)
        self.rec_id.currentIndexChanged.connect(self.toggle_conf)
        self.set_rec_conf_from_button.clicked.connect(self.load_rec_conf_dir)


    def load_tab(self, load_dir):
        """
        It verifies given configuration file, reads the parameters, and fills out the window.
        Parameters
        ----------
        conf : str
            configuration file (config_rec)
        Returns
        -------
        nothing
        """
        conf = os.path.join(load_dir, 'conf', 'config_rec')
        self.load_tab_common(conf)


    def load_tab_common(self, conf, update_rec_choice=True):
        if not os.path.isfile(conf):
            msg_window('info: the load directory does not contain ' + conf + ' file')
            return
        if not ver.ver_config_rec(conf):
            msg_window('please check configuration file ' + conf + '. Cannot parse, ')
            return
        try:
            conf_map = ut.read_config(conf)
        except Exception as e:
            msg_window('please check configuration file ' + conf + '. Cannot parse, ' + str(e))
            return

        # this will update the configuration choices by reading configuration files names
        # do not update when doing toggle
        if update_rec_choice:
            self.update_rec_configs_choice()

        try:
            self.device.setText(str(conf_map.device).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.reconstructions.setText(str(conf_map.reconstructions).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.alg_seq.setText(str(conf_map.algorithm_sequence).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.beta.setText(str(conf_map.beta).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.support_area.setText(str(conf_map.support_area).replace(" ", ""))
        except AttributeError:
            pass

        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].init_config(conf_map)

        self.notify()


    def clear_conf(self):
        self.device.setText('')
        self.reconstructions.setText('')
        self.alg_seq.setText('')
        self.beta.setText('')
        self.support_area.setText('')
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
            conf_map['reconstructions'] = str(self.reconstructions.text())
        if len(self.device.text()) > 0:
            conf_map['device'] = str(self.device.text()).replace('\n','')
        if len(self.alg_seq.text()) > 0:
            conf_map['algorithm_sequence'] = str(self.alg_seq.text()).replace('\n','')
        if len(self.beta.text()) > 0:
            conf_map['beta'] = str(self.beta.text())
        if len(self.support_area.text()) > 0:
            conf_map['support_area'] = str(self.support_area.text()).replace('\n','')
        if self.cont.isChecked():
            if len(self.cont_dir_button.text().strip()) > 0:
                conf_map['continue_dir'] = '"' + str(self.cont_dir_button.text()).strip() + '"'

        for feat_id in self.features.feature_dir:
            self.features.feature_dir[feat_id].add_config(conf_map)

        return conf_map


    def save_conf(self):
        conf_map = self.get_rec_config()
        if len(conf_map) > 0:
            write_conf(conf_map, os.path.join(self.main_win.experiment_dir, 'conf'), 'config_rec')


    def toggle_cont(self):
        """
        Invoked when the 'cont' checkbox is selected, indicating this reconstruction is continuation.
        Parameters
        ----------
        layout : QFormLayout
            a layout to add the continue dir

        Returns
        -------
        nothing
        """
        if self.cont.isChecked():
            self.cont_dir_label.show()
            self.cont_dir_button.show()
        else:
            self.cont_dir_label.hide()
            self.cont_dir_button.hide()


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
        cont_dir = select_dir(os.getcwd())
        if cont_dir is not None:
            self.cont_dir_button.setStyleSheet("Text-align:left")
            self.cont_dir_button.setText(cont_dir)
        else:
            self.cont_dir_button.setText('')


    def add_rec_conf(self):
        id, ok = QInputDialog.getText(self, '', "enter configuration id")
        if id in self.rec_ids:
            msg_window('the ' + id + ' is alredy used')
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
        conf_file = os.path.join(self.main_win.experiment_dir, 'conf', 'config_rec')
        new_conf_file = os.path.join(self.main_win.experiment_dir, 'conf', id + '_config_rec')
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
        # save the configuration file before updating the incoming config
        if self.old_conf_id == '':
            conf_file = 'config_rec'
        else:
            conf_file = self.old_conf_id + '_config_rec'

        conf_map = self.get_rec_config()
        conf_dir = os.path.join(self.main_win.experiment_dir, 'conf')

        if write_conf(conf_map, conf_dir, conf_file):
            if str(self.rec_id.currentText()) == 'main':
                self.old_conf_id = ''
            else:
                self.old_conf_id = str(self.rec_id.currentText())
        else:
            msg_window('configuration  ' + conf_file + ' was not saved')
        # if a config file corresponding to the rec id exists, load it
        # otherwise read from base configuration and load
        if self.old_conf_id == '':
            conf_file = os.path.join(conf_dir, 'config_rec')
        else:
            conf_file = os.path.join(conf_dir, self.old_conf_id + '_config_rec')

        if os.path.isfile(conf_file):
            # load the tab with new configuration, but do not update rec choices
            self.load_tab_common(conf_file, False)
        else:
            self.load_tab_common(os.path.join(conf_dir, 'config_rec'), False)
        self.notify()


    def load_rec_conf_dir(self):
        """
        It display a select dialog for user to select a configuration file. When selected, the parameters from that file will be loaded to the window.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        rec_file = select_file(os.getcwd())
        if rec_file is not None:
            self.load_tab_common(rec_file)
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
        import run_rec as run_rc

        if not self.main_win.is_exp_exists():
            msg_window('the experiment has not been created yet')
        elif not self.main_win.is_exp_set():
            msg_window('the experiment has changed, pres "set experiment" button')
        else:
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
                if self.old_conf_id == '':
                    conf_file = 'config_rec'
                    conf_id = None
                else:
                    conf_file = self.old_conf_id + '_config_rec'
                    conf_id = self.old_conf_id

                conf_map = self.get_rec_config()
                conf_dir = os.path.join(self.main_win.experiment_dir, 'conf')

                if write_conf(conf_map, conf_dir, conf_file):
                    run_rc.manage_reconstruction(str(self.proc.currentText()), self.main_win.experiment_dir, conf_id)
                    self.notify()
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
            self.device.setText('(0,1)')
            self.alg_seq.setText('((3,("ER",20),("HIO",180)),(1,("ER",20)))')
            self.beta.setText('.9')
            self.support_area.setText('(0.5, 0.5, 0.5)')
            self.cont.setChecked(False)


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
        # fill out the config_id choice bar by reading configuration files names
        self.rec_ids = []
        for file in os.listdir(os.path.join(self.main_win.experiment_dir, 'conf')):
            if file.endswith('_config_rec'):
                self.rec_ids.append(file[0:len(file)-len('_config_rec')])
        if len(self.rec_ids) > 0:
            self.rec_id.addItems(self.rec_ids)
            self.rec_id.show()


    def notify(self):
        generations = 0
        if self.features.feature_dir['GA'].active.isChecked():
            generations = int(self.features.feature_dir['GA'].generations.text())
        if len(self.reconstructions.text()) > 0:
            rec_no = int(self.reconstructions.text())
        else:
            rec_no = 1
        self.tabs.notify(**{'rec_id':self.old_conf_id, 'generations':generations, 'rec_no':rec_no})


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
        try:
            gens = conf_map.generations
            self.active.setChecked(True)
            self.generations.setText(str(gens).replace(" ", ""))
        except AttributeError:
            self.active.setChecked(False)
            return
        try:
            self.metrics.setText(str(conf_map.ga_metrics).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.breed_modes.setText(str(conf_map.ga_breed_modes).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.removes.setText(str(conf_map.ga_cullings).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.ga_support_thresholds.setText(str(conf_map.ga_support_thresholds).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.ga_support_sigmas.setText(str(conf_map.ga_support_sigmas).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.lr_sigmas.setText(str(conf_map.ga_low_resolution_sigmas).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.gen_pcdi_start.setText(str(conf_map.gen_pcdi_start).replace(" ", ""))
        except AttributeError:
            pass


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
        self.generations = QLineEdit()
        layout.addRow("generations", self.generations)
        self.metrics = QLineEdit()
        layout.addRow("fitness metrics", self.metrics)
        self.breed_modes = QLineEdit()
        layout.addRow("breed modes", self.breed_modes)
        self.removes = QLineEdit()
        layout.addRow("cullings", self.removes)
        self.ga_support_thresholds = QLineEdit()
        layout.addRow("after breed support thresholds", self.ga_support_thresholds)
        self.ga_support_sigmas = QLineEdit()
        layout.addRow("after breed support sigmas", self.ga_support_sigmas)
        self.lr_sigmas = QLineEdit()
        layout.addRow("low resolution sigmas", self.lr_sigmas)
        self.gen_pcdi_start = QLineEdit()
        layout.addRow("gen to start pcdi", self.gen_pcdi_start)


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
        self.generations.setText('5')
        self.metrics.setText('("chi","chi","area","chi","sharpness")')
        self.breed_modes.setText('("sqrt_ab","sqrt_ab","avg_ab","max_ab_pa","sqrt_ab")')
        self.removes.setText('(2,2,1)')
        self.ga_support_thresholds.setText('(.1,.1,.1,.1,.1)')
        self.ga_support_sigmas.setText('(1.0,1.0,1.0,1.0)')
        self.lr_sigmas.setText('(2.0,1.5)')
        self.gen_pcdi_start.setText('3')
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
        conf_map['generations'] = str(self.generations.text())
        conf_map['ga_metrics'] = str(self.metrics.text()).replace('\n','')
        conf_map['ga_breed_modes'] = str(self.breed_modes.text()).replace('\n','')
        conf_map['ga_cullings'] = str(self.removes.text()).replace('\n','')
        conf_map['ga_support_thresholds'] = str(self.ga_support_thresholds.text()).replace('\n','')
        conf_map['ga_support_sigmas'] = str(self.ga_support_sigmas.text()).replace('\n','')
        conf_map['ga_low_resolution_sigmas'] = str(self.lr_sigmas.text()).replace('\n','')
        conf_map['gen_pcdi_start'] = str(self.gen_pcdi_start.text())


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
        try:
            triggers = conf_map.resolution_trigger
            self.active.setChecked(True)
            self.res_triggers.setText(str(triggers).replace(" ", ""))
        except AttributeError:
            self.active.setChecked(False)
            return
        try:
            self.sigma_range.setText(str(conf_map.iter_res_sigma_range).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.det_range.setText(str(conf_map.iter_res_det_range).replace(" ", ""))
        except AttributeError:
            pass


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
        self.res_triggers = QLineEdit()
        layout.addRow("low resolution triggers", self.res_triggers)
        self.sigma_range = QLineEdit()
        layout.addRow("sigma range", self.sigma_range)
        self.det_range = QLineEdit()
        layout.addRow("det range", self.det_range)


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
        self.res_triggers.setText('(0, 1, 320)')
        self.sigma_range.setText('(2.0)')
        self.det_range.setText('(.7)')


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
        conf_map['resolution_trigger'] = str(self.res_triggers.text()).replace('\n','')
        conf_map['iter_res_sigma_range'] = str(self.sigma_range.text()).replace('\n','')
        conf_map['iter_res_det_range'] = str(self.det_range.text()).replace('\n','')


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
        try:
            triggers = conf_map.shrink_wrap_trigger
            self.active.setChecked(True)
            self.shrink_wrap_triggers.setText(str(triggers).replace(" ", ""))
        except AttributeError:
            self.active.setChecked(False)
            return
        try:
            self.shrink_wrap_type.setText(str(conf_map.shrink_wrap_type).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.threshold.setText(str(conf_map.support_threshold).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.sigma.setText(str(conf_map.support_sigma).replace(" ", ""))
        except AttributeError:
            pass


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
        self.threshold = QLineEdit()
        layout.addRow("threshold", self.threshold)
        self.sigma = QLineEdit()
        layout.addRow("sigma", self.sigma)


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
        self.shrink_wrap_triggers.setText('(1,1)')
        self.shrink_wrap_type.setText('GAUSS')
        self.sigma.setText('1.0')
        self.threshold.setText('0.1')


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
            conf_map['shrink_wrap_trigger'] = str(self.shrink_wrap_triggers.text()).replace('\n','')
        if len(self.shrink_wrap_type.text()) > 0:
            conf_map['shrink_wrap_type'] = '"' + str(self.shrink_wrap_type.text()) + '"'
        if len(self.threshold.text()) > 0:
            conf_map['support_threshold'] = str(self.threshold.text())
        if len(self.sigma.text()) > 0:
            conf_map['support_sigma'] = str(self.sigma.text())


class phase_support(Feature):
    """
    This class encapsulates phase support feature.
    """
    def __init__(self):
        super(phase_support, self).__init__()
        self.id = 'phase support'


    def init_config(self, conf_map):
        """
        This function sets phase support feature's parameters to parameters in dictionary and displays in the window.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        try:
            triggers = conf_map.phase_support_trigger
            self.active.setChecked(True)
            self.phase_triggers.setText(str(triggers).replace(" ", ""))
        except AttributeError:
            self.active.setChecked(False)
            return
        try:
            self.phase_min.setText(str(conf_map.phase_min).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.phase_max.setText(str(conf_map.phase_max).replace(" ", ""))
        except AttributeError:
            pass


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
        layout.addRow("phase support triggers", self.phase_triggers)
        self.phase_min = QLineEdit()
        layout.addRow("phase minimum", self.phase_min)
        self.phase_max = QLineEdit()
        layout.addRow("phase maximum", self.phase_max)


    def rec_default(self):
        """
        This function sets phase support feature's parameters to hardcoded default values.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        self.phase_triggers.setText('(0,1,320)')
        self.phase_min.setText('-1.57')
        self.phase_max.setText('1.57')


    def add_feat_conf(self, conf_map):
        """
        This function adds phase support feature's parameters to dictionary.
        Parameters
        ----------
        conf_map : dict
            contains parameters for reconstruction
        Returns
        -------
        nothing
        """
        conf_map['phase_support_trigger'] = str(self.phase_triggers.text()).replace('\n','')
        conf_map['phase_min'] = str(self.phase_min.text())
        conf_map['phase_max'] = str(self.phase_max.text())


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
        try:
            triggers = conf_map.pcdi_trigger
            self.active.setChecked(True)
            self.pcdi_triggers.setText(str(triggers).replace(" ", ""))
        except AttributeError:
            self.active.setChecked(False)
            return
        try:
            self.pcdi_type.setText(str(conf_map.partial_coherence_type).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.pcdi_iter.setText(str(conf_map.partial_coherence_iteration_num).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.pcdi_normalize.setText(str(conf_map.partial_coherence_normalize).replace(" ", ""))
        except AttributeError:
            pass
        try:
            self.pcdi_roi.setText(str(conf_map.partial_coherence_roi).replace(" ", ""))
        except AttributeError:
            pass


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
        self.pcdi_triggers = QLineEdit()
        layout.addRow("pcdi triggers", self.pcdi_triggers)
        self.pcdi_type = QLineEdit()
        layout.addRow("partial coherence algorithm", self.pcdi_type)
        self.pcdi_iter = QLineEdit()
        layout.addRow("pcdi iteration number", self.pcdi_iter)
        self.pcdi_normalize = QLineEdit()
        layout.addRow("normalize", self.pcdi_normalize)
        self.pcdi_roi = QLineEdit()
        layout.addRow("pcdi kernel area", self.pcdi_roi)


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
        self.pcdi_triggers.setText('(50,50)')
        self.pcdi_type.setText('LUCY')
        self.pcdi_iter.setText('20')
        self.pcdi_normalize.setText('true')
        self.pcdi_roi.setText('(16, 16, 16)')


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
        conf_map['pcdi_trigger'] = str(self.pcdi_triggers.text()).replace('\n','')
        conf_map['partial_coherence_type'] = '"' + str(self.pcdi_type.text()) + '"'
        conf_map['partial_coherence_iteration_num'] = str(self.pcdi_iter.text())
        conf_map['partial_coherence_normalize'] = str(self.pcdi_normalize.text())
        conf_map['partial_coherence_roi'] = str(self.pcdi_roi.text()).replace('\n','')


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
        try:
            triggers = conf_map.twin_trigger
            self.active.setChecked(True)
            self.twin_triggers.setText(str(triggers).replace(" ", ""))
        except AttributeError:
            self.active.setChecked(False)
            return
        try:
            self.twin_halves.setText(str(conf_map.twin_halves).replace(" ", ""))
        except AttributeError:
            pass


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
        self.twin_triggers.setText('(2)')
        self.twin_halves.setText('(0,0)')


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
        conf_map['twin_trigger'] = str(self.twin_triggers.text()).replace('\n','')
        conf_map['twin_halves'] = str(self.twin_halves.text()).replace('\n','')


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
        try:
            triggers = conf_map.average_trigger
            self.active.setChecked(True)
            self.average_triggers.setText(str(triggers).replace(" ", ""))
        except AttributeError:
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
        self.average_triggers.setText('(-50,1)')


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
        conf_map['average_trigger'] = str(self.average_triggers.text()).replace('\n','')


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
        try:
            triggers = conf_map.progress_trigger
            self.active.setChecked(True)
            self.progress_triggers.setText(str(triggers).replace(" ", ""))
        except AttributeError:
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
        self.progress_triggers.setText('(0,20)')


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
        conf_map['progress_trigger'] = str(self.progress_triggers.text()).replace('\n','')


class Features(QWidget):
    """
    This class is composition of all feature classes.
    """
    def __init__(self, tab, layout):
        """
        Constructor, creates all concrete feature objects, and displays in window.
        """
        super(Features, self).__init__()
        feature_ids = ['GA', 'low resolution', 'shrink wrap', 'phase support', 'pcdi', 'twin', 'average', 'progress']
        self.leftlist = QListWidget()
        self.feature_dir = {'GA' : GA(),
                            'low resolution' : low_resolution(),
                            'shrink wrap' : shrink_wrap(),
                            'phase support' : phase_support(),
                            'pcdi' : pcdi(),
                            'twin' : twin(),
                            'average' : average(),
                            'progress' : progress()}
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


def main(args):
    """
    Starts GUI application.
    """
    app = QApplication(args)
    ex = cdi_gui()
    ex.set_args(args)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv[1:])
