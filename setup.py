from setuptools import find_packages, setup
from typing import List

# This constant represents the special line "-e ."
# "-e ." tells pip to install the project in "editable" mode.
# It appears inside requirements.txt when you use local development mode.
HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    '''
    Reads the requirements.txt file and returns a clean list of dependencies.

    Why this function is needed:
    ----------------------------
    - requirements.txt may contain extra characters like "\n" or "-e ."
    - We need to clean the list before sending it to install_requires
    - This keeps the installation process smooth and bug-free

    :param file_path: path to the requirements file
    :type file_path: str
    :return: list of cleaned requirement strings
    :rtype: List[str]
    '''

    requirements = []

    # Open the requirements file and read all lines
    with open("requirements.txt") as file_obj:
        requirements = file_obj.readlines()

        # Remove newline characters from each line:
        # Example: "pandas\n" becomes "pandas"
        [req.replace("\n", "") for req in requirements]

        # Remove "-e ." if present, since it's not an actual package.
        # "-e ." is only used for local editable installations.
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



# ---------------------------------------------------------------------------
# SETUP FUNCTION
# ---------------------------------------------------------------------------
# setup() is the core function that turns your folder into an installable
# Python package. It tells pip:
# - the name of your project
# - the version
# - who the author is
# - what packages to include
# - what dependencies to install
# ---------------------------------------------------------------------------
setup(
    name="mlprojects",       # The name of your package
    version="0.0.1",         # Version number (increment when updating)
    author="Syfer",          # Your name
    author_email="tkeddy5@gmail.com",  # Contact information

    # find_packages() automatically discovers all modules inside your project
    # that contain an __init__.py file.
    # Example: src/, src/components/, src/utils/
    packages=find_packages(),

    # These are the external packages required for your project to run.
    # They are taken directly from requirements.txt using the function above.
    install_requires=get_requirements("requirements.txt")
)
