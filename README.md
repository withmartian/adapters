# adapters

Package to easily route among different LLMs

## Installing

1. Setup Python `3.11.6`
2. Install [Poetry](https://python-poetry.org/docs/#installation)
3. Install dependencies - `poetry install`
4. Install pre-commit hooks `poetry run pre-commit install`
5. Run commands via poetry `poetry run pytest`

## Setting up Pre-commit

Make sure to install the pre-commit plugins before contributing to the lib. `poetry run pre-commit install`. This should enable pre-commit plugin, and should be done after the installation.

To run pre-commit manually: `poetry run pre-commit run --all-files`

## Environment variables

Please note that the callers of this package would be responsible for setting up all the correct environment variables.
For testing we use an env file. refer to .env-example as a template.
This file needs to be copied to a .env file and then updated with the correct values.
Other repo using this package should do the same and setup the correct environment variables.
In this case you can just copy all the missing values from the .env-example into your own project and make sure you assign values.

## Running Test

This package depends on python 3.11, so you must have such a python interpreter.

- Make sure you installed [poetry and dependencies](# Installing)
- Run test: `poetry run pytest`

## Disabling models

You can also disable specific models by setting the environment variable `ADAPTER_DISABLED_MODELS` to either a single, or multiple models:
`ADAPTER_DISABLED_MODELS=model1` or `ADAPTER_DISABLED_MODELS=model1,model2`.
This will not prevent the factory from creating these models, however get supported models will return a list without the disabled models.
So any automatic iteration will only use none disabled models

### How to use adapters

To access the adapters, you can call:

The new way suggested firs to create an adapter instance, rather than have one created on the fly.
This make things more specific, removes some the abstraction and let's you interact directly withe the adapter.
Also allowing you to re-use the adapter for subsequent calls.
The new way uses the `AdapterFactor` to create adapters and use them.

1. Create an adapter instance:
   `adapter1 = AdapterFactory.get_adapter(<adapter_name>)` - - `adapter_name` - a martain adapter name (normally formatted as `provider/vendor/model_nameg`), for full names call `AdapterFactory.get_supported_models()`
2. All adapter have a `.convert_to_input` that converts a Prompt or Conversation to the model specific input. (look at tests for examples)
3. Use the newly created adapter instance `adapter1.execute_async` or `adapter1.execute_async`

### Getting list of all models

`AdapterFactory.get_supported_models()`, `get_supported_models()` are identical ways to get the list of models, the second one being syntactic sugar for the first call.

## Adding new Models / Contributions / Code Changes the lib

Feel free to develop your own models. Adding models is very simple.

1. Create a new model class, if it's an abstract base class please put in the abstractAdapters, if it's a concreteClass for a specific model, put it in the concreteAdapters.
2. Add you new class to the concreteAdapters `__init__.py` file.
3. That's it, you can now access this model, the AdapterFactory will load this adapter class automatically for you.
4. Add the relevant tests:

   4.1. if you're adding an OpenAI or Anthropic, make sure to add the new models to the list of models and token costs lists in the tests/utils.py file so they will get tested
   4.2 if it's new tests make sure to add them in the relevant folder in the same tree as the source tree
   4.3 If you run tests that make network requests make sure to decorate the tests with @pytest.mark.vcr, this will create a network cassette file for your test

5. Run the test as described before `poetry run pytest`
6. Make sure to also check in the newly created cassette yaml file, as test in circle have no network access.
7. Verify tests pass and send a PR for review.
8. If you need re-create cassette files ( change the pytest.ini or run pytest with --record-mode=rewrite)
   **note that some models are accessable only from the US, in such cases to re-generate cassette files you might need to ask someone in the use to run, or use ssh into a us based machine**
