# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 09:17:00 2018

@author: dkang
"""

from music21 import converter
import music21_helper
import os

directory_name = 'C:/Users/dkang/Documents/cs229p/Classical'
directory = os.fsencode(directory_name)

counter = 0
total_vec = []
for file in os.listdir(directory)[492:723]:
    filename = os.fsdecode(file)
    holder = converter.parse(directory_name+'/'+filename)
    total_vec.append(music21_helper.organize_song_midi_length(holder))
    
    counter+= 1
    print(counter)
    
with open('classical_notes.pkl', 'wb') as handle:
        pickle.dump(total_vec, handle, protocol= pickle.HIGHEST_PROTOCOL )

#########################################################
#Get all individual note durations
note_duration = []
for song in total_vec:
    for note in song:
        holder = note.numpy()
        if not holder[12] in note_duration:
            note_duration.append(holder[12])
        if holder[12] > 10:
            print(holder)

note_duration = note_duration[0:63]

with open('note_duration_list.pkl', 'wb') as handle:
        pickle.dump(note_duration, handle, protocol= pickle.HIGHEST_PROTOCOL )
        
#Get all individual num of time signature
num_time_sig = []
for song in total_vec:
    for note in song:
        holder = note.numpy()
        if not holder[14] in num_time_sig:
            num_time_sig.append(holder[14])
num_time_sig.sort()

den_time_sig = []
for song in total_vec:
    for note in song:
        holder = note.numpy()
        if not holder[15] in den_time_sig:
            den_time_sig.append(holder[15])
den_time_sig.sort()

#Get all key signature
key_sig = []
for song in total_vec:
    for note in song:
        holder = note.numpy()
        if not holder[16] in key_sig:
            key_sig.append(holder[16])
key_sig.sort()
