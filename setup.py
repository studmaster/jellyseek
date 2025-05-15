from setuptools import setup, find_packages

setup(
    name="jellyseek",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.32.0",
        "python-dotenv>=1.0.1",
        "langchain-ollama",
        "chromadb"
    ],
    entry_points={
        'console_scripts': [
            'jellyseek=jellyseek.__main__:main',
        ],
    },
)