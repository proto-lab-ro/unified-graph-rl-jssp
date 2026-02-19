# Type definitions
Operation = tuple[int, int]  # (machine_id, duration)
Job = list[Operation]  # List of operations in sequence
F3X3_INSTANCE = """
# This is a 3x3 test instance
3 3
0 3 1 2 2 2
0 2 2 1 1 4
1 4 2 3 0 3
"""
# Standard test instances as text
FT06_INSTANCE = """
2  1  0  3  1  6  3  7  5  3  4  6
1  8  2  5  4 10  5 10  0 10  3  4
2  5  3  4  5  8  0  9  1  1  4  7
1  5  0  5  2  5  3  3  4  8  5  9
2  9  1  3  4  5  5  4  0  3  3  1
1  3  3  3  5  9  0 10  4  4  2  1
"""
