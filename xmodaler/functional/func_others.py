# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""

def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]