# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['mltools']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib>=3.5.2,<4.0.0', 'numpy>=1.22.0,<2.0.0', 'pandas>=2.0.0,<3.0.0']

setup_kwargs = {
    'name': 'mltools',
    'version': '0.1.42',
    'description': 'Machine Learning functions aiming to help in the Machine Learning course at ICAI',
    'long_description': '# mltools\n\nMachine Learning functions aiming to help in the Machine Learning course at ICAI\n\n## Installation\n\n```bash\n$ pip install mltools\n```\n\n## Usage\n\n- TODO\n\n## Contributing\n\nInterested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.\n\n## License\n\n`mltools` was created by Jaime Pizarroso. It is licensed under the terms of the GNU General Public License v3.0 license.\n\n## Credits\n\n`mltools` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).\n',
    'author': 'Jaime Pizarroso',
    'author_email': None,
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
