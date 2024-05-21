from collections import defaultdict
import pandas as pd
from adapters import AdapterFactory


def create_provider_models_table():

    supported_models = AdapterFactory.get_supported_models()

    models_by_provider = defaultdict(list)

    def format_supports(k: str) -> str:
        parts = k.split('_')
        return ' '.join([p.capitalize() for p in parts[1:]])

    for model in supported_models:
        models_by_provider[ model.__class__.__name__[:-5]].append({
            "Model Name": model.name,
            **{format_supports(k): v for k, v in model.model_dump().items() if k.startswith("supports_")}
        })

    providers = set(models_by_provider.keys())
    ordered_providers = ['OpenAI', 'Anthropic', 'Together']

    # Render the markdown
    rendered_markdown = ""
    while providers:
        if ordered_providers:
            provider = ordered_providers.pop(0)
            providers.remove(provider)
        else:
            provider = providers.pop()
        provider_models = models_by_provider[provider]
        rendered_markdown += f"#### {provider}\n\n"
        rendered_markdown += pd.DataFrame(provider_models).replace({True: '✓', False: '✗'}).to_markdown(index=False)
        rendered_markdown += '\n\n\n'
    return rendered_markdown


def update_supported_models_file(filepath):
    with open(filepath, 'r', encoding='utf8') as file:
        content = file.read()

    # Remove the models section
    content = content.split("####", 1)[0]

    # Add in the updated models section
    content += create_provider_models_table()

    with open(filepath, 'w', encoding='utf8') as file:
        file.write(content)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str)
    args = parser.parse_args()
    update_supported_models_file(args.filepath)
