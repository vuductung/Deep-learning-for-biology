import unittest
from dl_biology.functions import aa_encoder


class TestAAEncoder(unittest.TestCase):
    """Unit tests for the aa_encoder function."""
    
    def test_single_amino_acid_string(self):
        """Test encoding a single amino acid as a string."""
        result = aa_encoder("A")
        self.assertEqual(result, [0])
    
    def test_single_amino_acid_list(self):
        """Test encoding a single amino acid as a list."""
        result = aa_encoder(["A"])
        self.assertEqual(result, [0])
    
    def test_multiple_amino_acids_string(self):
        """Test encoding multiple amino acids as a string."""
        result = aa_encoder("ACD")
        self.assertEqual(result, [0, 1, 2])
    
    def test_multiple_amino_acids_list(self):
        """Test encoding multiple amino acids as a list."""
        result = aa_encoder(["A", "C", "D"])
        self.assertEqual(result, [0, 1, 2])
    
    def test_all_amino_acids_string(self):
        """Test encoding all 20 standard amino acids as a string."""
        all_aa = "ACDEFGHIKLMNPQRSTVWY"
        result = aa_encoder(all_aa)
        expected = list(range(20))
        self.assertEqual(result, expected)
    
    def test_all_amino_acids_list(self):
        """Test encoding all 20 standard amino acids as a list."""
        all_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        result = aa_encoder(all_aa)
        expected = list(range(20))
        self.assertEqual(result, expected)
    
    def test_empty_string(self):
        """Test encoding an empty string."""
        result = aa_encoder("")
        self.assertEqual(result, [])
    
    def test_empty_list(self):
        """Test encoding an empty list."""
        result = aa_encoder([])
        self.assertEqual(result, [])
    
    def test_repeated_amino_acids_string(self):
        """Test encoding repeated amino acids in a string."""
        result = aa_encoder("AAA")
        self.assertEqual(result, [0, 0, 0])
    
    def test_repeated_amino_acids_list(self):
        """Test encoding repeated amino acids in a list."""
        result = aa_encoder(["A", "A", "A"])
        self.assertEqual(result, [0, 0, 0])
    
    def test_correct_indices_for_specific_amino_acids(self):
        """Test that specific amino acids map to correct indices."""
        # A should be 0, Y should be 19
        result_a = aa_encoder("A")
        result_y = aa_encoder("Y")
        self.assertEqual(result_a, [0])
        self.assertEqual(result_y, [19])
    
    def test_long_protein_sequence_string(self):
        """Test encoding a longer protein sequence as a string."""
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 10  # 200 amino acids
        result = aa_encoder(sequence)
        self.assertEqual(len(result), 200)
        # Check first 20 are correct
        self.assertEqual(result[:20], list(range(20)))
    
    def test_order_preservation_string(self):
        """Test that the order of amino acids is preserved for strings."""
        result = aa_encoder("YVCDA")
        # Y=19, V=17, C=1, D=2, A=0
        self.assertEqual(result, [19, 17, 1, 2, 0])
    
    def test_order_preservation_list(self):
        """Test that the order of amino acids is preserved for lists."""
        result = aa_encoder(["Y", "V", "C", "D", "A"])
        # Y=19, V=17, C=1, D=2, A=0
        self.assertEqual(result, [19, 17, 1, 2, 0])
    
    def test_return_type_is_list_for_string_input(self):
        """Test that the return type is a list when input is a string."""
        result = aa_encoder("ACD")
        self.assertIsInstance(result, list)
    
    def test_return_type_is_list_for_list_input(self):
        """Test that the return type is a list when input is a list."""
        result = aa_encoder(["A", "C", "D"])
        self.assertIsInstance(result, list)
    
    def test_return_values_are_integers_string_input(self):
        """Test that all returned values are integers for string input."""
        result = aa_encoder("ACD")
        self.assertTrue(all(isinstance(x, int) for x in result))
    
    def test_return_values_are_integers_list_input(self):
        """Test that all returned values are integers for list input."""
        result = aa_encoder(["A", "C", "D"])
        self.assertTrue(all(isinstance(x, int) for x in result))
    
    def test_invalid_amino_acid_string_raises_key_error(self):
        """Test that invalid amino acid in string raises KeyError."""
        with self.assertRaises(KeyError):
            aa_encoder("ABC123")
    
    def test_invalid_amino_acid_list_raises_key_error(self):
        """Test that invalid amino acid in list raises KeyError."""
        with self.assertRaises(KeyError):
            aa_encoder(["A", "B", "X"])
    
    def test_lowercase_amino_acid_string_raises_key_error(self):
        """Test that lowercase amino acids in string raise KeyError."""
        with self.assertRaises(KeyError):
            aa_encoder("acd")
    
    def test_lowercase_amino_acid_list_raises_key_error(self):
        """Test that lowercase amino acids in list raise KeyError."""
        with self.assertRaises(KeyError):
            aa_encoder(["a", "c", "d"])
    
    def test_special_characters_raise_key_error(self):
        """Test that special characters raise KeyError."""
        with self.assertRaises(KeyError):
            aa_encoder("A-C-D")
    
    def test_whitespace_raises_key_error(self):
        """Test that whitespace in sequence raises KeyError."""
        with self.assertRaises(KeyError):
            aa_encoder("A C D")
    
    def test_numeric_string_raises_key_error(self):
        """Test that numeric characters raise KeyError."""
        with self.assertRaises(KeyError):
            aa_encoder("123")


if __name__ == "__main__":
    unittest.main()

