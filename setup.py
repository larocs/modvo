from setuptools import setup, find_packages

setup(
    name='modvo',
    version='0.0.0',    
    description='A modular Visual Odometry pipeline',
    url='',
    author='Hudson Bruno',
    author_email='hudson.bruno@ic.unicamp.br',
    license="GPLv3",
    packages=find_packages(include=['modvo','modvo.*',]),
    install_requires=['numpy',                     
                      ],

     classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
)