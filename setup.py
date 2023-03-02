from setuptools import setup, find_packages

setup(
    name='nfn',
    version='0.1.0',
    description='Neural Functional Network (NFN) layers',
    url='https://github.com/AllanYangZhou/nfn',
    author='Allan Zhou',
    author_email='ayz@cs.stanford.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['einops', 'torch'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ]
)