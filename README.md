# cohere-ui
User files

This repository is submodule of cohere project and contains supplementary scripts to the cohere_core tools https://github.com/advancedPhotonSource/cohere. The cohere-ui package is beamline specific and supports the following beamlines: aps_1ide, aps_34idc, esrf_id01, Petra3_P10.

After installing this repository run installation:

    pip install -e .
    
Content:
1. A directory cohere_ui containing user scripts. Refer to documentation at https://cohere.readthedocs.io/en/latest/how_to_use.html for instruction on how to use the scripts.
2. A directory cohere-defaults containing configuration files listing all parameters. The config_instr and config_prep list parameters used by aps_34idc beamline. The beamline specific configuration is in cohere_ui/beamlines/<beamline> directory, where <beamline> is a specific beamline. Refer to documentation at https://cohere.readthedocs.io/en/latest/configuration.html for detail description of all supported parameters.
3. A directory example_data containing experiment data necessary to conduct analysis.
4. A directory example_workspace directory with example experiment space containing configuration files. The example experiment is provided to users to learn the tools by trying on the real data and real environment.
5. A hpc_scripts directory containing scripts running the tools on Polaris.
