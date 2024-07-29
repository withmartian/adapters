# Adapters Package Documentation

## Overview

The Adapters package facilitates communication between different language model APIs by providing a unified interface for interaction. This ensures ease of use and flexibility in integrating multiple models from various providers.

The package can be installed an used via pip:
```
pip install martian-adapters
```

## Getting Started

### Prerequisites

- Python version: 3.11.6
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation

```bash
poetry install
poetry run pre-commit install
```

### Setting Up Pre-commit

Pre-commit hooks help maintain code quality and standards. Install them with the following command:

```bash
poetry run pre-commit install
```

To run pre-commit manually:

```bash
poetry run pre-commit run --all-files
```

### Semantic Versioning

For versioning we follow [Semantic Versioning](https://semver.org)

### Environment Configuration

The package requires certain environment variables to be set by the users:

- Copy `.env-example` to `.env` and populate it with appropriate values.

### Running Tests

Ensure Python 3.11 is used:

```bash
poetry run pytest
```

## Quickstart

```python
from adapters import AdapterFactory, Prompt

adapter = AdapterFactory.get_adapter_by_path("openai/openai/gpt-4o-mini")

prompt = Prompt("Who is your favorite Martian?")
conversation = adapter.convert_to_input(prompt)

adapter.execute_sync(conversation)
```

Adapter paths follows the format `provider/vendor/model_name`. Use `AdapterFactory.get_supported_models()` to retrieve all supported models. For a given model, `model.get_path()` returns the adapter path.

## Contributing

### Adding New Models

1. **Existing Providers:**
   Add new models to the `SUPPORTED_MODELS` array if the provider is already supported.

2. **New Providers:**
   - If the provider follows the OpenAI format, model integration is straightforward. See the "_Together_" provider class as an example.
   - For providers with different schemas, see the "_Anthropic_" provider class for guidance.

### Development Steps

1. **Add the Provider and Model:** Update `provider_adapters/__init__.py` and test files accordingly.
2. **Write Tests:** Add tests in the relevant directories. Use `@pytest.mark.vcr` for tests making network requests.
3. **Run Tests:**

   ```bash
   poetry run pytest
   ```

4. **Check-in Cassette Files:** Include any new cassette YAML files in your commit.
5. **Send a Pull Request:** Ensure all tests pass before requesting a review.

### Re-creating Cassette Files

Use the `--record-mode=rewrite` option with pytest to update cassette files.

## Additional Notes

Some models may only be accessible from specific locations (e.g., the U.S.). In such cases, running tests might require access to a U.S.-based server.

This documentation provides a streamlined approach to using and contributing to the Adapters package, emphasizing practical steps and clear examples.
