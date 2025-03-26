import setuptools

setuptools.setup(
      name='cohere_scripts',
      author = 'Barbara Frosik, Ross Harder',
      author_email = 'bfrosik@anl.gov',
      url='https://github.com/advancedPhotonSource/cohere/cohere-ui',
      version='4.2.0',
      packages=['cohere_scripts', 'cohere_scripts.inner_scripts'],
      install_requires=[
                        'mayavi',
                        'scikit-image',
                        'xrayutilities',
                        'vtk==9.3.1',
                        'scipy==1.14.1',
                        ],
      classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
      ],
)
