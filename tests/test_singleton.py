from src.etl.processor import get_vlm_parser, _GLOBAL_VLM_PARSER
import src.etl.processor as processor

def test_singleton_logic():
    print("Testing VLM Singleton Logic...")
    
    # In this environment, we can't load the real VLM
    # But we can mock the VLMParser class to test the processor's logic
    class MockParser:
        def __init__(self, model_name):
            self.model_name = model_name
            print(f"MOCK: Loading model {model_name}")

    # Patch the VLMParser in the processor module
    original_vlm = processor.VLMParser
    processor.VLMParser = MockParser
    
    try:
        # 1. First call should initialize
        p1 = get_vlm_parser(model_name="test-model")
        assert p1 is not None
        assert processor._GLOBAL_VLM_PARSER is p1
        
        # 2. Second call should return same instance
        p2 = get_vlm_parser(model_name="test-model")
        assert p2 is p1
        print("✅ Singleton confirmed: Both calls returned the same instance.")
        
    finally:
        # Cleanup
        processor.VLMParser = original_vlm
        processor._GLOBAL_VLM_PARSER = None

if __name__ == "__main__":
    test_singleton_logic()
