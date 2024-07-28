
import jellyfish as j
import fuzzywuzzy as f

def levenshtein_distance(str1,str2):
    score = f.fuzz.ratio(str1, str2)/100
    return score    
