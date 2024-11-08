from adapters.adapter_factory import AdapterFactory

# Load supported models
supported_models = AdapterFactory.get_supported_models()

# Define emojis for boolean values
tick = "✅"
cross = "❌"

table = """# Supported Models

This table provides an overview of the supported models, including their vendor, provider, cost details, and capabilities.

| Model | Vendor | Provider | Prompt Cost | Completion Cost | Request Cost | Context Length | Completion Length | User | Repeating Roles | Streaming | Vision | Tools | Supports N | System | Multiple Systems | Empty Content | Tool Choice | Tool Choice Required | JSON Output | JSON Content | Last Assistant | First Assistant | Temperature | Only System | Only Assistant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""

# Add model rows with formatted values
for model in supported_models:
    table += (
        f"| {model.name} | {model.vendor_name} | {model.provider_name} | ${model.cost.prompt} | ${model.cost.completion} | ${model.cost.request} | {model.context_length} | {model.completion_length or 'N/A'} | "
        f"{tick if model.supports_user else cross} | {tick if model.supports_repeating_roles else cross} | {tick if model.supports_streaming else cross} | {tick if model.supports_vision else cross} | "
        f"{tick if model.supports_tools else cross} | {tick if model.supports_n else cross} | {tick if model.supports_system else cross} | {tick if model.supports_multiple_system else cross} | "
        f"{tick if model.supports_empty_content else cross} | {tick if model.supports_tool_choice else cross} | {tick if model.supports_tool_choice_required else cross} | "
        f"{tick if model.supports_json_output else cross} | {tick if model.supports_json_content else cross} | {tick if model.supports_last_assistant else cross} | "
        f"{tick if model.supports_first_assistant else cross} | {tick if model.supports_temperature else cross} | {tick if model.supports_only_system else cross} | {tick if model.supports_only_assistant else cross} |\n"
    )

# Save the markdown table to file
with open("docs/index.html", "w") as f:
    f.write(table)
