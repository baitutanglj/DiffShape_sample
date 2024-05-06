from setuptools import setup, find_packages

reqs=[
    ]

setup(
    name='DiffShape_sample',
    version='0.0.1',
    url=None,
    author='HongMing Chen’s research group',
    author_email='',
    description='DiffShape_sample',
    packages=find_packages(exclude=["archives", "configs"]),
    install_requires=reqs
)
