# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:59:43 2018

@author: dkang
"""

from music21 import converter
import os
import music21
from music21 import *
from music21 import roman
from music21 import interval
from music21 import chord
import pickle
import fractions
import numpy as np
import torch
num_time_sig = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 12.0]
den_time_sig = [2.0, 4.0, 8.0, 16.0]
note_duration = [0.0833, 0.1667, 0.25, 0.3333, 0.4167, 0.5, 0.5833, 0.6667, 0.75, 0.8333, 0.9167, 1.0, 1.0833, 
                 1.1667, 1.25, 1.3333, 1.4167, 1.5, 1.5833, 1.6667, 1.75, 1.8333, 1.9167, 2.0, 2.0833, 2.1667, 2.25, 2.3333, 
                 2.4167, 2.5, 2.5833, 2.6667, 2.75, 2.8333, 2.9167, 3.0, 3.1667, 3.25, 3.3333, 3.4167, 3.5, 3.6667, 
                 3.75, 4.0, 4.25, 4.3333, 4.5, 4.6667, 4.75, 5.0, 5.25, 5.4167, 5.5, 5.75, 6.0, 6.5, 6.75, 6.8333, 
                 7.0, 7.75, 8.0, 9.5, 10.0]
key_sig = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
directory_name = 'C:/Users/dkang/Documents/cs229p/Generation'
directory = os.fsencode(directory_name)



counter = 0
total_vec = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    holder = converter.parse(directory_name+'/'+filename)
    holderc = holder.chordify()
    
    return_vec = []
    for n in holderc.notesAndRests[0:100]:
        note_vec = torch.zeros(219)
        if type(n) is music21.note.Rest:
            if type(n.quarterLength) is fractions.Fraction:
                if np.round(n.quarterLength.numerator/n.quarterLength.denominator,4) > 10:
                    note_vec[191] = 1
                else:
                    note_vec[128+note_duration.index(np.round(n.quarterLength.numerator/n.quarterLength.denominator,4))] = 1
            else:
                if n.quarterLength > 10:
                    note_vec[191] = 1
                else:
                    note_vec[128+note_duration.index(n.quarterLength)] = 1
            note_vec[192+num_time_sig.index(holderc.timeSignature.numerator)] = 1
            note_vec[200+den_time_sig.index(holderc.timeSignature.denominator)] = 1
            note_vec[204+key_sig.index(holderc.keySignature.sharps)] = 1
        else:
            for p in n:
                note_vec[p.pitch.midi] = 1
                if type(n.quarterLength) is fractions.Fraction:
                    if np.round(n.quarterLength.numerator/n.quarterLength.denominator,4) > 10:
                        note_vec[191] = 1
                    else:
                        note_vec[128+note_duration.index(np.round(n.quarterLength.numerator/n.quarterLength.denominator,4))] = 1
                else:
                    if n.quarterLength > 10:
                        note_vec[191] = 1
                    else:
                        note_vec[128+note_duration.index(n.quarterLength)] = 1
                if not holderc.timeSignature:
                    note_vec[194] = 1
                    note_vec[201] = 1
                else:
                    note_vec[192+num_time_sig.index(holderc.timeSignature.numerator)] = 1
                    note_vec[200+den_time_sig.index(holderc.timeSignature.denominator)] = 1
                if not holderc.keySignature:
                    note_vec[211] = 1
                else:
                    note_vec[204+key_sig.index(holderc.keySignature.sharps)] = 1
        return_vec.append(note_vec)
    
    total_vec.append(return_vec)
    
    counter+= 1
    print(counter)
    
with open('genlist.pkl', 'wb') as handle:
        pickle.dump(total_vec, handle, protocol= pickle.HIGHEST_PROTOCOL )

