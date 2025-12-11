from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]: 
    '''
    Docstring for get_requirements
    
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: List[str]
    '''
    requirements=[]
    with open("requirements.txt") as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="mlprojects",
    version="0.0.1",
    author="Syfer",
    author_email="tkeddy5@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)