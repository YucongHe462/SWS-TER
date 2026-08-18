from setuptools import find_packages, setup


setup(
    name='sws-ter',
    version='1.0.0',
    description='Sparse weakly semi-supervised tri-evidence recovery for PolSAR ships',
    python_requires='>=3.9',
    packages=find_packages(include=(
        'projects*', 'semi_mmrotate*', 'mmdet*', 'mmrotate*')),
    include_package_data=True,
    license='Apache-2.0',
)

