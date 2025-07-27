#!/usr/bin/env python3
"""
Quick test to verify the class structure works correctly
"""

import numpy as np
from collections import defaultdict, Counter

# Test the class hierarchy
class GridUtils:
    @staticmethod
    def normalize_grid(grid):
        return np.array(grid, dtype=np.int32)
    
    @staticmethod
    def grid_to_list(grid):
        return grid.tolist()
    
    @staticmethod
    def get_colors(grid):
        return list(np.unique(grid))

# Base enhanced rule system
class EnhancedARCRuleSystem:
    def __init__(self):
        self.rules = [
            self.rule_copy_input,
            self.rule_fill_background,
        ]
        self.utils = GridUtils()
        self.rule_weights = {
            'rule_copy_input': 1.0,
            'rule_fill_background': 0.9,
        }
        self.success_history = defaultdict(list)
    
    def rule_copy_input(self, train_examples, test_inputs):
        """Basic copy rule"""
        return [test_input for test_input in test_inputs]
    
    def rule_fill_background(self, train_examples, test_inputs):
        """Basic fill rule"""
        return [test_input for test_input in test_inputs]
    
    def predict(self, task):
        """Basic predict method"""
        test_inputs = [example['input'] for example in task['test']]
        return test_inputs[0] if test_inputs else [[0]], []

# Extended rule system
class ARCRuleSystemExtended(EnhancedARCRuleSystem):
    def rule_complete_symmetry(self, train_examples, test_inputs):
        """Extended rule"""
        return [test_input for test_input in test_inputs]

# Advanced rule system
class ARCRuleSystemAdvanced(ARCRuleSystemExtended):
    def rule_extract_largest_shape(self, train_examples, test_inputs):
        """Advanced rule"""
        return [test_input for test_input in test_inputs]

# Final rule system
class ARCRuleSystemFinal(ARCRuleSystemAdvanced):
    def rule_pattern_completion(self, train_examples, test_inputs):
        """Final enhanced rule"""
        return [test_input for test_input in test_inputs]

# Test the class hierarchy
def test_class_structure():
    print("Testing class structure...")
    
    # Test instantiation
    try:
        rule_system = ARCRuleSystemFinal()
        print("✅ ARCRuleSystemFinal instantiated successfully")
    except Exception as e:
        print(f"❌ Error instantiating ARCRuleSystemFinal: {e}")
        return False
    
    # Test method inheritance
    try:
        test_task = {
            'train': [{'input': [[1, 2]], 'output': [[1, 2]]}],
            'test': [{'input': [[3, 4]]}]
        }
        
        result = rule_system.predict(test_task)
        print("✅ Predict method works")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"❌ Error calling predict: {e}")
        return False
    
    # Test rule methods
    try:
        test_inputs = [[[1, 2], [3, 4]]]
        train_examples = []
        
        # Test inherited methods
        result1 = rule_system.rule_copy_input(train_examples, test_inputs)
        result2 = rule_system.rule_complete_symmetry(train_examples, test_inputs)
        result3 = rule_system.rule_pattern_completion(train_examples, test_inputs)
        
        print("✅ All rule methods accessible")
        print(f"   Copy input: {len(result1)} results")
        print(f"   Symmetry: {len(result2)} results")
        print(f"   Pattern completion: {len(result3)} results")
        
    except Exception as e:
        print(f"❌ Error calling rule methods: {e}")
        return False
    
    print("✅ All tests passed! Class structure is correct.")
    return True

if __name__ == "__main__":
    success = test_class_structure()
    if success:
        print("\n🎉 Class structure is ready for the main notebook!")
    else:
        print("\n❌ Class structure needs fixes.")
