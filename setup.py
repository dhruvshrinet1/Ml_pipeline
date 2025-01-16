from setuptools import find_packages,setup

HYPHEN_E_DOT = "-e ."
def get_requirements(fillpath:str)->list[str]:

    requirements = []
    with open(fillpath,'r') as f:
        requirements = f.readlines()
        requirements = [line.strip() for line in requirements if line.strip()]  
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements    
        

setup(
    name='mlproject',
    version='1.0',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'my_script = my_package.my_script:main'
        ]
    }
)