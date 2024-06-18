from collections import defaultdict

import pandas as pd  # type: ignore[import-untyped]

from adapters import AdapterFactory

NAME_ONLY_PROVIDERS = ["OpenAI", "Anthropic", "Together"]
ORDERED_PROVIDERS = ["OpenAI", "Anthropic", "Together"]
DROPPED_COLUMNS = {"supports_functions", "supports_n"}


def create_provider_models_table():
    supported_models = AdapterFactory.get_supported_models()
    models_by_provider = defaultdict(list)

    def format_supports(k: str) -> str:
        # formats "supports_streaming" to "Streaing"
        parts = k.split("_")
        return " ".join([p.capitalize() for p in parts[1:]])

    for model in supported_models:
        provider = model.__class__.__name__[:-5]
        models_by_provider[provider].append(
            {
                "Model": model.name
                if provider in NAME_ONLY_PROVIDERS
                else model.get_path(),
                **{
                    format_supports(k): v
                    for k, v in model.model_dump().items()
                    if k.startswith("supports_") and k not in DROPPED_COLUMNS
                },
            }
        )

    providers = set(models_by_provider.keys())

    # Render the markdown
    rendered_markdown = ""
    ordered_providers = ORDERED_PROVIDERS[:]
    while providers:
        if ordered_providers:
            provider = ordered_providers.pop(0)
            providers.remove(provider)
        else:
            provider = providers.pop()
        provider_models = models_by_provider[provider]
        rendered_markdown += f"### {provider}\n\n"
        rendered_markdown += (
            pd.DataFrame(provider_models)
            .set_index("Model")
            .replace({True: "✅", False: "❌"})
            .to_markdown()
        )
        rendered_markdown += "\n\n\n"
    return rendered_markdown


def update_supported_models_file(filepath):
    with open(filepath, "r", encoding="utf8") as file:
        content = file.read()

    # Remove the models section
    content = content.split(f"### {ORDERED_PROVIDERS[0]}")[0]
    while content[-1] == "#":
        content = content[:-1]
    # Add in the updated models section
    content += create_provider_models_table()

    with open(filepath, "w", encoding="utf8") as file:
        file.write(content)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("filepath", type=str)
    args = parser.parse_args()
    update_supported_models_file(args.filepath)
