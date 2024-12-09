from adapters.adapter_factory import AdapterFactory

# Load supported models
supported_models = AdapterFactory.get_supported_models()

# Define emojis for boolean values
tick = "✅"
cross = "❌"

table = """# Supported Models

This table provides an overview of the supported models, including their vendor, provider, cost details, and capabilities.

|     Model     | Vendor | Provider | Prompt $ | Completion $ | Request $ | Context | Completion | User | Repeating Roles | Streaming | Vision | Tools | Supports N | System | Multiple Systems | Empty Content | Tool Choice | Tool Choice Required | JSON Output | JSON Content | Last Assistant | First Assistant | Temperature | Only System | Only Assistant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""

# Add model rows with formatted values
for model in supported_models:
    table += (
        f"| {model.name} | {model.vendor_name} | {model.provider_name} | "
        f"${model.cost.prompt} | ${model.cost.completion} | ${model.cost.request} | "
        f"{model.context_length} | {model.completion_length or 'N/A'} | "
        f"{tick if model.can_user else cross} | "
        f"{tick if model.can_repeating_roles else cross} | "
        f"{tick if model.supports_streaming else cross} | "
        f"{tick if model.supports_vision else cross} | "
        f"{tick if model.supports_tools else cross} | "
        f"{tick if model.supports_n else cross} | "
        f"{tick if model.can_system else cross} | "
        f"{tick if model.can_system_multiple else cross} | "
        f"{tick if model.can_empty_content else cross} | "
        f"{tick if model.supports_tools_choice else cross} | "
        f"{tick if model.supports_tools_choice_required else cross} | "
        f"{tick if model.supports_json_output else cross} | "
        f"{tick if model.supports_json_content else cross} | "
        f"{tick if model.can_assistant_last else cross} | "
        f"{tick if model.can_assistant_first else cross} | "
        f"{tick if model.can_temperature else cross} | "
        f"{tick if model.can_system_only else cross} | "
        f"{tick if model.can_assistant_only else cross} |\n"
    )

# Save the markdown table to file
with open("docs/index.md", "w") as f:
    f.write(table)
