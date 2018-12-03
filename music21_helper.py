# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:03:27 2018

@author: dkang
"""

from music21 import *
from music21 import roman
from music21 import interval
from music21 import chord
import pickle
import fractions
import numpy as np
import torch

s = converter.parse('C:/Users/dkang/Downloads/simple_music/Double/A la Claire Fontaine.3a14b35116f79bbe1167cece3e0b7a1b.mid')
sChord = s.chordify()

num_time_sig = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 12.0]
den_time_sig = [2.0, 4.0, 8.0, 16.0]
note_duration = [0.0833, 0.1667, 0.25, 0.3333, 0.4167, 0.5, 0.5833, 0.6667, 0.75, 0.8333, 0.9167, 1.0, 1.0833, 
                 1.1667, 1.25, 1.3333, 1.4167, 1.5, 1.5833, 1.6667, 1.75, 1.8333, 1.9167, 2.0, 2.0833, 2.1667, 2.25, 2.3333, 
                 2.4167, 2.5, 2.5833, 2.6667, 2.75, 2.8333, 2.9167, 3.0, 3.1667, 3.25, 3.3333, 3.4167, 3.5, 3.6667, 
                 3.75, 4.0, 4.25, 4.3333, 4.5, 4.6667, 4.75, 5.0, 5.25, 5.4167, 5.5, 5.75, 6.0, 6.5, 6.75, 6.8333, 
                 7.0, 7.75, 8.0, 9.5, 10.0]
#with open('note_duration_list.pkl', 'rb') as handle:
#    note_duration = pickle.load(handle)
key_sig = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

def organize_song_midi(song):
    """
    Takes in a song and return of vector of tensor of notes played in that chord along with length of note
    
    0-127: Midi notes
    128: Length of chord
    129: Beat Strength
    130: Numerator of Time Signature
    131: Denominator of Time Signature
    132: Key Signature
    """
    chordi = song.chordify()
    
    return_vec = []
    for n in chordi.notesAndRests:
        note_vec = torch.zeros(132)
        if typ(n) is note.Rest:
            if type(n.quarterLength) is fractions.Fraction:
                note_vec[128] = np.round(n.quarterLength.numerator/n.quarterLength.denominator,4)
            else:
                note_vec[128] = n.quarterLength
            note_vec[129] = n.beatStrength
            note_vec[130] = chordi.timeSignature.numerator
            note_vec[131] = chordi.timeSignature.denominator
            note_vec[132] = chordi.keySignature.sharps
        for p in n:
            note_vec[p.pitch.midi] = 1
            if type(n.quarterLength) is fractions.Fraction:
                note_vec[128] = np.round(n.quarterLength.numerator/n.quarterLength.denominator,4)
            else:
                note_vec[128] = n.quarterLength
            note_vec[129] = n.beatStrength
            note_vec[130] = chordi.timeSignature.numerator
            note_vec[131] = chordi.timeSignature.denominator
            note_vec[132] = chordi.keySignature.sharps
        return_vec.append(note_vec)
    
    return return_vec

def organize_song_pc(song):
    """
    Takes in a song and return of vector of tensor of notes played in that chord along with length of note
    
    0-11: Pitch Class
    12: Length of chord
    13: Beat Strength
    14: Numerator of Time Signature
    15: Denominator of Time Signature
    16: Key Signature
    """
    chordi = song.chordify()
    
    return_vec = []
    for n in chordi.notesAndRests:
        note_vec = torch.zeros(17)
        if type(n) is note.Rest:
            if type(n.quarterLength) is fractions.Fraction:
                note_vec[12] = np.round(n.quarterLength.numerator/n.quarterLength.denominator,4)
            else:
                note_vec[12] = n.quarterLength
            note_vec[13] = n.beatStrength
            note_vec[14] = chordi.timeSignature.numerator
            note_vec[15] = chordi.timeSignature.denominator
            note_vec[16] = chordi.keySignature.sharps
        else:
            for p in n:
                note_vec[p.pitch.pitchClass] = 1
                if type(n.quarterLength) is fractions.Fraction:
                    note_vec[12] = np.round(n.quarterLength.numerator/n.quarterLength.denominator,4)
                else:
                    note_vec[12] = n.quarterLength
                note_vec[13] = n.beatStrength
                note_vec[14] = chordi.timeSignature.numerator
                note_vec[15] = chordi.timeSignature.denominator
                note_vec[16] = chordi.keySignature.sharps
        return_vec.append(note_vec)
    
    return return_vec

def organize_song_midi_length(song):
    """
    Takes in a song and return of vector of tensor of notes played in that chord along with length of note
    
    0-127: Midi notes
    128-191: Note durations
    192-199: Numerator of Time Signature
    200-203: Denominator of Time Signature
    204-218: Key Signature
    """
    chordi = song.chordify()
    return_vec = []
    
    for n in chordi.notesAndRests:
        note_vec = torch.zeros(219)
        if type(n) is note.Rest:
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
            note_vec[192+num_time_sig.index(chordi.timeSignature.numerator)] = 1
            note_vec[200+den_time_sig.index(chordi.timeSignature.denominator)] = 1
            note_vec[204+key_sig.index(chordi.keySignature.sharps)] = 1
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
                note_vec[192+num_time_sig.index(chordi.timeSignature.numerator)] = 1
                note_vec[200+den_time_sig.index(chordi.timeSignature.denominator)] = 1
                note_vec[204+key_sig.index(chordi.keySignature.sharps)] = 1
        return_vec.append(note_vec)
    
    return return_vec
    
