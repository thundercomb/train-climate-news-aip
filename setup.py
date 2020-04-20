from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'six==1.13.0',
    'gpt_2_simple',
    'google-resumable-media==0.3.2',
    'google-cloud-bigquery==1.17.0',
    'google-cloud-storage==1.20.0'
]

setup(
    name='train-climate-news-aip',
    version='0.1.0',
    description='Package for Climate News - Train Climate News with AI Platform',
    author='thundercomb',
    url='https://github.com/thundercomb/ingest-ncei-wind',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(exclude=('test'))
)
