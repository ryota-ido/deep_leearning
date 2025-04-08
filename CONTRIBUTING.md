# Contributing to the Machine Learning Framework

## Welcome!

We're thrilled that you're interested in contributing to our Machine Learning Framework. This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others. Harassment and discriminatory behavior are not tolerated.

## Getting Started

### 1. Fork the Repository

1. Fork the main repository on GitHub
2. Clone your fork locally
```bash
git clone https://github.com/your-username/ml-framework.git
cd ml-framework
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development Workflow

### Branch Naming Convention
- `feature/`: New features
- `bugfix/`: Bug fixes
- `docs/`: Documentation updates
- `refactor/`: Code refactoring

Example:
```bash
git checkout -b feature/add-new-preprocessing-method
```

### Coding Standards

#### Language and Documentation
- All code and comments must be in English
- Use type hints and docstrings for all public methods
- Follow PEP 8 style guidelines

#### Docstring Guidelines
Each public method and class should have a docstring that includes:
- A clear description of its purpose
- Args section detailing parameters
- Returns section explaining the return value
- Raises section for potential exceptions

Example:
```python
def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data.

    Args:
        data: Input DataFrame to be processed

    Returns:
        Preprocessed DataFrame with transformations applied

    Raises:
        ValueError: If input data is invalid
    """
```

### Testing

- Write unit tests for new features
- Ensure 100% code coverage for new code
- Run tests before submitting a pull request
```bash
pytest tests/
pytest --cov=src tests/
```

### Code Reviews
- All contributions require a review from a maintainer
- Be open to feedback and constructive criticism
- Discuss proposed changes before extensive implementation

## Submitting a Pull Request

1. Ensure your code follows the project's coding standards
2. Write tests for new features
3. Update documentation
4. Squash commits into a single, coherent commit
5. Write a clear, descriptive pull request title and description

### Pull Request Template
```markdown
## Description
[Provide a detailed description of the changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## How Tested
[Describe the tests that you ran to verify changes]

## Checklist
- [ ] I have performed a self-review of my code
- [ ] I have added tests that prove my fix/feature works
- [ ] My changes generate no new warnings
- [ ] I have updated the documentation
```

## Reporting Bugs

### Before Submitting a Bug Report
- Check existing issues to avoid duplicates
- Gather information:
  - Python version
  - Framework version
  - Detailed error description
  - Reproducible code sample

### Submitting a Bug Report
Use GitHub Issues with a clear title and description. Include:
- Detailed steps to reproduce
- Expected behavior
- Actual behavior
- Environment details
- Stack trace or error message

## Feature Requests

1. Check existing issues and discussions
2. Provide a clear and detailed explanation
3. Explain the motivation and use case
4. Be prepared to discuss and refine the proposal

## Development Setup

### Pre-commit Hooks
We use pre-commit hooks to ensure code quality:
```bash
pip install pre-commit
pre-commit install
```

### Continuous Integration
- All pull requests must pass CI checks
- Code coverage must not decrease

## Communication

- Join our Slack/Discord channel
- Use GitHub Discussions for broader topics
- File issues for specific problems or feature requests

## Recognition

Contributors will be recognized in the project's README and CONTRIBUTORS file.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have any questions, please open an issue or contact the maintainers.

Happy coding! 🚀
