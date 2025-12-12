#!/usr/bin/python
'''
typical commands to run:  

python calc_Rg.py -p /binfl/lv70806/DrZverev/all_in_flow/posdip_reduced_data/ --p_number 2 --p_idx 1
'''

import argparse
import collections
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import scipy.special as sp


def create_parser():
    parser = argparse.ArgumentParser(description=('script for calculation '
                                                  'gyradius'
                                                  ))
    parser.add_argument('-p',
                        '--path',
                        type=Path,
                        default=Path('.'),
                        help = 'path to folders with pos_dip_folders')


    parser.add_argument('--p_number',
                        type=int,
                        default=0,
                        help='if p_number > 0 then list of files is splitted on the p_number parts')

    parser.add_argument('--p_idx',
                        type=int,
                        default=0,
                        help='if p_number > 0 then only list_of_files[p_idx] is processed')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='debug mode')
    return parser


def calc_Rg (particles_pos):
    # particle_pos is array (matrix N on 3) of particle's position, type is np.array
    coord = np.zeros(particles_pos.shape)
    for i in range(3):
        coord[:,i] = particles_pos[:,i] - np.mean(particles_pos[:,i])
    sqr_array = np.sum(coord**2, axis=1)    
    return np.sqrt(np.mean(sqr_array))


def print_elapsed_time(start):
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("Elapsed time  = {0:2.0f}h:{1:2.0f}m:{2:2.0f}s".format(h, m, s))

if __name__ == '__main__':
    start = time.time()
    parser = create_parser()
    cmd_arguments = parser.parse_args()
    print(cmd_arguments)

    raise NotImplementedError
                    
    print_elapsed_time(start)
