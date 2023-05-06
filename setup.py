from setuptools import find_packages, setup
from typing import List

Hyphen_E_dot = "-e ."
def get_requirements(file_path : str) -> List[str]:
    """
    Get requirements from file path.
    
    
    Input: file_path
    Output: requirements
    
    """


    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.read().splitlines()

        if Hyphen_E_dot in requirements:
            requirements.remove(Hyphen_E_dot)
    return requirements

    
# get_requirements('requirements.txt') # test function.


setup(
    name = 'DL_package',
    version = '0.0.1',
    author = 'Amine MEKKI',
    author_email = 'amine.mekki@mines-ales.org',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    
)