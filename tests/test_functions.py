import unittest

import torch

from dl_biology.helper import aa_encoder
from dl_biology.model import PositionalEncoder, TransformerEncoderRt


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


class TestTransformerEncoderRt(unittest.TestCase):
    """Unit tests for the TransformerEncoderRt class."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 21  # 20 amino acids + padding
        self.d_model = 128
        self.dim_feedforward = 512
        self.nhead = 8
        self.dropout = 0.1
        self.max_len = 100
        self.num_layers = 2

    def test_transformer_initialization(self):
        """Test that TransformerEncoderRt initializes correctly."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        # Check that the model was created successfully
        self.assertIsInstance(transformer, TransformerEncoderRt)
        self.assertEqual(transformer.d_model, self.d_model)

    def test_forward_pass_output_shape(self):
        """Test that forward pass returns correct output shape."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        batch_size = 4
        seq_len = 10

        # Create test input
        x = torch.randint(1, self.vocab_size, (batch_size, seq_len))  # Skip padding token (0)
        lengths = torch.tensor([seq_len] * batch_size)

        output = transformer(x, lengths)

        # Check output shape
        self.assertEqual(output.shape, (batch_size,))
        self.assertIsInstance(output, torch.Tensor)

    def test_different_batch_sizes(self):
        """Test TransformerEncoderRt with different batch sizes."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        seq_len = 15
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            x = torch.randint(1, self.vocab_size, (batch_size, seq_len))
            lengths = torch.tensor([seq_len] * batch_size)

            output = transformer(x, lengths)

            # Check output shape
            self.assertEqual(output.shape, (batch_size,))

    def test_different_sequence_lengths(self):
        """Test TransformerEncoderRt with different sequence lengths."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        batch_size = 3
        seq_lengths = [5, 10, 20]

        for seq_len in seq_lengths:
            x = torch.randint(1, self.vocab_size, (batch_size, seq_len))
            lengths = torch.tensor([seq_len] * batch_size)

            output = transformer(x, lengths)

            # Check output shape
            self.assertEqual(output.shape, (batch_size,))

    def test_variable_sequence_lengths(self):
        """Test TransformerEncoderRt with variable sequence lengths in the same batch."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        batch_size = 4
        max_seq_len = 20
        seq_lengths = [5, 10, 15, 20]

        # Create input with padding
        x = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        lengths = torch.tensor(seq_lengths)

        for i, seq_len in enumerate(seq_lengths):
            x[i, :seq_len] = torch.randint(1, self.vocab_size, (seq_len,))

        output = transformer(x, lengths)

        # Check output shape
        self.assertEqual(output.shape, (batch_size,))

    def test_with_sample_data(self):
        """Test TransformerEncoderRt with realistic sample data."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        # Create sample peptide sequences (using amino acid indices)
        # A=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, K=9, L=10, M=11, N=12, P=13, Q=14, R=15, S=16, T=17, V=18, W=19, Y=20
        sample_sequences = [
            [1, 2, 3, 4, 5],  # AC DEF (length 5)
            [6, 7, 8, 9, 10, 11, 12, 13],  # GHIKLMNP (length 8)
            [14, 15, 16, 17, 18, 19, 20],  # QRSTVWY (length 7)
        ]

        batch_size = len(sample_sequences)
        max_seq_len = max(len(seq) for seq in sample_sequences)
        lengths = torch.tensor([len(seq) for seq in sample_sequences])

        # Pad sequences to same length
        x = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        for i, seq in enumerate(sample_sequences):
            x[i, : len(seq)] = torch.tensor(seq)

        output = transformer(x, lengths)

        # Check output shape
        self.assertEqual(output.shape, (batch_size,))

        # Check that outputs are not NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_gradient_flow(self):
        """Test that gradients can flow through the model."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        batch_size = 2
        seq_len = 10

        x = torch.randint(1, self.vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([seq_len] * batch_size)

        output = transformer(x, lengths)
        loss = output.sum()
        loss.backward()

        # Check that gradients were computed for model parameters
        has_gradients = any(p.grad is not None for p in transformer.parameters())
        self.assertTrue(has_gradients)

    def test_model_parameters(self):
        """Test that the model has the expected number of parameters."""
        transformer = TransformerEncoderRt(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            max_len=self.max_len,
            num_layers=self.num_layers,
        )

        # Count parameters
        total_params = sum(p.numel() for p in transformer.parameters())
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

        # Check that we have parameters
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All parameters should be trainable


if __name__ == "__main__":
    unittest.main()
