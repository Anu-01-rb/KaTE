from setuptools import find_packages, setup
setup(
    name='KaTE',
    packages=find_packages(include=['KaTE']),
    version='0.1.0',
    description='The AI and DS Tools',
    author='Anurak',
    license='MIT',
    install_requires=[
        "numpy",
        "opencv-python",
        "pythainlp",
        "wordcloud",
        "matplotlib", 
    ],
)
