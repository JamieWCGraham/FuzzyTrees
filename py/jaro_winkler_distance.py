
import jellyfish as j

def jaro_winkler_distance(str1,str2):
    score = j.jaro_winkler_similarity(str1,str2)
    return score