import math

def convert_to_decimal(base, seq):
    decimal = 0
    for i in range(len(seq)):
        value = int(seq[i])*math.pow(int(base), len(seq)-i-1)
        decimal += value
    return decimal
        
    
def calculate_diff(base1, seq1, base2, seq2):
    """ calculate decimal difference between two sequences given their bases"""
    dec1 = convert_to_decimal(base1, seq1)
    dec2 = convert_to_decimal(base2, seq2)
    diff = abs(dec1-dec2)
    return diff