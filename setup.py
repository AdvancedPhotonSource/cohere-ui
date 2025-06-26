import os
import setuptools

cur_dir = os.getcwd().replace(os.sep, '/')

with open('example_workspace/scan_54/conf/config', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config', 'w') as file:
    file.write(filedata)

with open('example_workspace/scan_54/conf/config_prep', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config_prep', 'w') as file:
    file.write(filedata)

with open('example_workspace/scan_54/conf/config_disp', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config_disp', 'w') as file:
    file.write(filedata)

with open('example_workspace/scan_54/conf/config_instr', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/scan_54/conf/config_instr', 'w') as file:
    file.write(filedata)

with open('example_workspace/esrf_exp_4/conf/config', 'r') as file :
    filedata = file.read()
filedata = filedata.replace('CUR_DIR', cur_dir)
with open('example_workspace/esrf_exp_4/conf/config', 'w') as file:
    file.write(filedata)

setuptools.setup(
      name='cohere_ui',
      author='Barbara Frosik, Ross Harder',
      author_email='bfrosik@anl.gov',
      url='https://github.com/advancedPhotonSource/cohere/cohere-ui',
      version='4.2.0',
      packages=['cohere_ui', 
                'cohere_ui.api', 
                'cohere_ui.beamlines.aps_1ide', 
                'cohere_ui.beamlines.aps_34idc', 
                'cohere_ui.beamlines.esrf_id01', 
                'cohere_ui.beamlines.Petra3_P10', 
                'cohere_ui.beamlines.simple'],
      install_requires=[
                         'pyqt5',
                         'scikit-image',
                         'xrayutilities',
                         'pyvista',
                         'scipy==1.14.1',
                         'notebook',
                         'gputil',
                        ],
      classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
      ],
)
