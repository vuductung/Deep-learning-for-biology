import unittest

import torch

from dl_biology.helper import aa_encoder
from dl_biology.model import PositionalEncoder


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
        all_aa = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
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


class TestPositionalEncoder(unittest.TestCase):
    """Unit tests for the PositionalEncoder class."""

    def test_pe_matrix_size(self):
        """Test that PositionalEncoder creates pe matrix with correct size (1, 20, 3)."""
        pe = PositionalEncoder(d_model=4, max_len=20, dropout=0.1)

        # Check that the positional encoding matrix has the correct size
        self.assertEqual(pe.pe.shape, (1, 20, 4))

        # Check that it's a tensor
        self.assertIsInstance(pe.pe, torch.Tensor)

    def test_forward_pass_output_size(self):
        """Test that forward pass returns tensor with correct size (5, 4, 3)."""
        pe = PositionalEncoder(d_model=6, max_len=20, dropout=0.1)
        x = torch.zeros(5, 4, 6)

        output = pe(x)

        # Check that the output has the correct size
        self.assertEqual(output.shape, (5, 4, 6))

        # Check that it's a tensor
        self.assertIsInstance(output, torch.Tensor)

    def test_pe_matrix_values_not_zero(self):
        """Test that the positional encoding matrix contains non-zero values."""
        pe = PositionalEncoder(d_model=6, max_len=20, dropout=0.1)

        # Check that the matrix is not all zeros (positional encoding should have values)
        self.assertFalse(torch.allclose(pe.pe, torch.zeros_like(pe.pe)))

    def test_different_input_sizes(self):
        """Test PositionalEncoder with different input sizes."""
        pe = PositionalEncoder(d_model=6, max_len=20, dropout=0.1)

        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 5, 6),  # batch=1, seq_len=5, d_model=3
            (2, 10, 6),  # batch=2, seq_len=10, d_model=3
            (3, 15, 6),  # batch=3, seq_len=15, d_model=3
        ]

        for batch_size, seq_len, d_model in test_cases:
            x = torch.zeros(batch_size, seq_len, d_model)
            output = pe(x)
            self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_dropout_behavior(self):
        """Test that dropout is applied in forward pass."""
        pe_no_dropout = PositionalEncoder(d_model=6, max_len=20, dropout=0.0)
        pe_with_dropout = PositionalEncoder(d_model=6, max_len=20, dropout=0.5)

        x = torch.zeros(5, 4, 6)

        # Set models to eval mode to disable dropout
        pe_no_dropout.eval()
        pe_with_dropout.eval()

        output_no_dropout = pe_no_dropout(x)
        output_with_dropout = pe_with_dropout(x)

        # In eval mode, outputs should be identical regardless of dropout rate
        self.assertTrue(torch.allclose(output_no_dropout, output_with_dropout))


if __name__ == "__main__":
    unittest.main()
