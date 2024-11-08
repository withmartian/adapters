from adapters.adapter_factory import AdapterFactory

supported_models = AdapterFactory.get_supported_models()

# Generate markdown table to render for all models and save it to a file

table = """| Model | Vendor | Provider | Prompt Cost | Completion Cost | Request Cost | Context Length | Completion Length | Supports User | Supports Repeating Roles | Supports Streaming | Supports Vision | Supports Tools | Supports N | Supports System | Supports Multiple Systems | Supports Empty Content | Supports Tool Choice | Supports Tool Choice Required | Supports JSON Output | Supports JSON Content | Supports Last Assistant | Supports First Assistant | Supports Temperature | Supports Only System | Supports Only Assistant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""

for model in supported_models:
    table += f"| {model.name} | {model.vendor_name} | {model.provider_name} | {model.cost.prompt} | {model.cost.completion} | {model.cost.request} | {model.context_length} | {model.completion_length or 'N/A'} | {model.supports_user} | {model.supports_repeating_roles} | {model.supports_streaming} | {model.supports_vision} | {model.supports_tools} | {model.supports_n} | {model.supports_system} | {model.supports_multiple_system} | {model.supports_empty_content} | {model.supports_tool_choice} | {model.supports_tool_choice_required} | {model.supports_json_output} | {model.supports_json_content} | {model.supports_last_assistant} | {model.supports_first_assistant} | {model.supports_temperature} | {model.supports_only_system} | {model.supports_only_assistant} |\n"

with open("docs/models.md", "w") as f:
    f.write(table)
