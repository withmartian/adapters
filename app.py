from adapters.adapter_factory import AdapterFactory

adapter = AdapterFactory.get_adapter_by_path("tests/adapters/utils/models/transformer")

adapter_response = adapter.execute_sync(adapter.convert_to_input("Hello, how are you?"))
