import tfplan


import os
from setuptools import setup, find_packages


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, 'r')
    return file.read()


setup(
    name='tf-plan',
    version=tfplan.__version__,
    author='Thiago P. Bueno',
    author_email='thiago.pbueno@gmail.com',
    description='Planning through backpropagation using TensorFlow.',
    long_description=read('README.md'),
    license='GNU General Public License v3.0',
    keywords=['planning', 'tensorflow', 'rddl', 'mdp'],
    url='https://github.com/thiagopbueno/tf-plan',
    packages=find_packages(),
    scripts=['scripts/tfplan'],
    install_requires=[
        'pyrddl',
        'tfrddlsim',
        'tensorflow',
        'tensorflow-tensorboard',
        'typing'
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
