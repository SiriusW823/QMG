"""Shared utilities for processing quantum measurement counts.

This module provides functions for reordering bitstrings in measurement counts
to match the classical register layout expected by the molecule generator.
"""

from typing import Dict, List


def reorder_counts(counts: Dict[str, int], order: List[int]) -> Dict[str, int]:
    """Reorder bitstrings in counts according to the measurement order.
    
    When qubits are measured out of order (relative to the classical register
    indices they should map to), this function rearranges the bits in each
    bitstring so that the final output matches the expected classical layout.
    
    Args:
        counts: Dictionary mapping bitstrings to their occurrence counts.
        order: List where order[i] indicates the classical bit index that
            the i-th measured qubit should be mapped to. If order is empty
            or already [0, 1, 2, ...], the counts are returned unchanged.
    
    Returns:
        Dictionary with bitstrings reordered according to the measurement order.
    
    Example:
        If order = [2, 0, 1], then a bitstring "ABC" (where A is bit 0 of the
        measurement result) becomes "BCA" (A goes to position 2, B goes to
        position 0, C goes to position 1).
    """
    if not order or order == list(range(len(order))):
        return counts
    
    reordered: Dict[str, int] = {}
    for bitstring, value in counts.items():
        bits = list(str(bitstring))
        if len(bits) != len(order):
            # If the bitstring length doesn't match, pass through unchanged
            key = str(bitstring)
            reordered[key] = reordered.get(key, 0) + value
            continue
        
        # Rearrange bits according to the order mapping
        arranged = ["0"] * len(order)
        for position, classical_index in enumerate(order):
            arranged[classical_index] = bits[position]
        
        key = "".join(arranged)
        reordered[key] = reordered.get(key, 0) + value
    
    return reordered
