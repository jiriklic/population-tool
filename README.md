# Tool name

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()
[![license](https://img.shields.io/github/license/OCHA-DAP/pa-aa-toolbox.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


## Usage

#### Requirements

The Python version currently used is 3.10. Please install all packages from
``requirements.txt``:

```shell
pip install -r requirements.txt
```

#### Run the app

Finally, open a terminal and run

```shell
streamlit run app/Home.py
```

A new browser window will open and you can start using the tool.

## Contributing

#### Pre-commit

All code is formatted according to
[black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest) guidelines. The repo is set-up to use [pre-commit](https://github.com/pre-commit/pre-commit). Please run ``pre-commit install`` the first time you are editing. Thereafter all commits will be checked against black and flake8 guidelines.

To check if your changes pass pre-commit without committing, run:

```shell
pre-commit run --all-files
```
