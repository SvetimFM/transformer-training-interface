# Contributing to Transformer PCN UI

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the existing style
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the requirements.txt if you add new dependencies
3. The PR will be merged once you have the sign-off of at least one maintainer

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.

## Report bugs using GitHub's [issues](https://github.com/yourusername/transformer-pcn-ui/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/transformer-pcn-ui/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Code Style

* Use Python 3.10+ features where appropriate
* Follow PEP 8 style guide
* Use type hints for function parameters and return values
* Add docstrings to all functions and classes
* Keep lines under 100 characters when possible
* Use meaningful variable names

## Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformer-pcn-ui.git
cd transformer-pcn-ui
```

2. Run the setup script:
```bash
python setup.py
```

3. Activate the virtual environment:
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

4. Install development dependencies:
```bash
pip install -r requirements-dev.txt  # If available
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

For specific test files:
```bash
pytest tests/test_pcn_manager.py
```

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Community

- Join our discussions in GitHub Issues
- Follow the project for updates
- Star the repository if you find it useful!

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Special contributors file (for significant contributions)

Thank you for contributing to make this project better!