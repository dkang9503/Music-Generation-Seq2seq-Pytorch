# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:54:10 2018

@author: dkang
"""

import numpy as np
import pickle
import torch
import music21
from encoderdecoder import seq2seq, EncoderRNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = EncoderRNN(219, 512)
decoder = DecoderRNN(512, 219)
    
model = seq2seq(encoder, decoder, device).to(device)

state = torch.load('modelstate.pth')
model.load_state_dict(state['state_dict'])


#Adjusting data so it can be put into model
with open('genlist.pkl', 'rb') as handle:
    data = pickle.load(handle)

data_list = [] #List of sequences for each song
for song in data:
    holder = torch.stack(song)
    holder = torch.cat((((torch.zeros(holder.shape[1])+2).unsqueeze(0)).to(device),holder))
    holder = torch.cat((holder, ((torch.zeros(holder.shape[1])+3).unsqueeze(0)).to(device)))
    data_list.append(holder)
        
#####################################Music generation###########################################
    
#Note Embedding References
num_time_sig = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 12.0]
den_time_sig = [2.0, 4.0, 8.0, 16.0]
note_duration = [0.0833, 0.1667, 0.25, 0.3333, 0.4167, 0.5, 0.5833, 0.6667, 0.75, 0.8333, 0.9167, 1.0, 1.0833, 
                 1.1667, 1.25, 1.3333, 1.4167, 1.5, 1.5833, 1.6667, 1.75, 1.8333, 1.9167, 2.0, 2.0833, 2.1667, 2.25, 2.3333, 
                 2.4167, 2.5, 2.5833, 2.6667, 2.75, 2.8333, 2.9167, 3.0, 3.1667, 3.25, 3.3333, 3.4167, 3.5, 3.6667, 
                 3.75, 4.0, 4.25, 4.3333, 4.5, 4.6667, 4.75, 5.0, 5.25, 5.4167, 5.5, 5.75, 6.0, 6.5, 6.75, 6.8333, 
                 7.0, 7.75, 8.0, 9.5, 10.0]
note_duration_frac = [1/12, 1/6, 1/4, 1/3, 5/12, 1/2, 7/12, 2/3, 3/4, 5/6, 11/12, 1.0, 13/12, 
                 7/6, 5/4, 4/3, 17/12, 1.5, 19/12, 5/3, 1.75, 22/12, 23/12, 2.0, 25/12, 7/6, 2.25, 7/3, 
                 29/12, 2.5, 31/12, 8/3, 2.75, 17/6, 35/12, 3.0, 19/6, 3.25, 10/3, 41/12, 3.5, 11/3, 
                 3.75, 4.0, 4.25, 13/3, 4.5, 14/3, 4.75, 5.0, 5.25, 65/12, 5.5, 5.75, 6.0, 6.5, 6.75, 41/6, 
                 7.0, 7.75, 8.0, 9.5, 10.0]
key_sig = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    
#Getting predictions of sequeunces
gen_list = [] #Sequence of predictions
for song in data_list:
    part1 = model(song, song, 0) #First 100 notes generated
    part2 = model(part1, part1, 0) #Second 100 notes generated

    #Converting probabilities to actual notes
    
    for idx, note in enumerate(part1): #for each note
        #Part1
        note_vec = part1[idx][0:128] > .5 #If probability is greater than .5, the note is played
        note_vec = note_vec.float() #because casted as int
        length_vec = torch.zeros(len(note_duration)+1) #Number of different types of durations
        length_vec[part1[idx][128:192].max(0)[1].item()] = 1 #Whichever length had highest probability is the one chosen
        num_vec = torch.zeros(len(num_time_sig)) #Number of different types of num key sig
        num_vec[part1[idx][192:200].max(0)[1].item()] = 1 #highest prob
        den_vec = torch.zeros(len(den_time_sig)) #Number of different types of den key sig
        den_vec[part1[idx][200:204].max(0)[1].item()] = 1 #highest prob
        key_sig = torch.zeros(len(key_sig)) #Key signature
        key_sig[part1[idx][204:219].max(0)[1].item()]=1 #highest prob
        part1[idx] = torch.cat((note_vec, length_vec, num_vec, den_vec, key_sig))
        
        #Part2
        note_vec1 = part2[idx][0:128] > .5 #If probability is greater than .5, the note is played
        note_vec1 = note_vec.float() #because casted as int
        length_vec1 = torch.zeros(len(note_duration)+1) #Number of different types of durations
        length_vec1[part2[idx][128:192].max(0)[1].item()] = 1 #Whichever length had highest probability is the one chosen
        num_vec1 = torch.zeros(len(num_time_sig)) #Number of different types of num key sig
        num_vec1[part2[idx][192:200].max(0)[1].item()] = 1 #highest prob
        den_vec1 = torch.zeros(len(den_time_sig)) #Number of different types of den key sig
        den_vec1[part2[idx][200:204].max(0)[1].item()] = 1 #highest prob
        key_sig1 = torch.zeros(len(key_sig)) #Key signature
        key_sig1[part2[idx][204:219].max(0)[1].item()]=1 #highest prob
        part2[idx] = torch.cat((note_vec, length_vec, num_vec, den_vec, key_sig))
    
    
    gen_list.append(torch.cat((part1[1:101,:], part2[1:101,:])).detach().numpy())
    
##########################Converting One hot to notes###########################################
stream_list = []
for song in gen_list:
    
    stream1 = music21.stream.Stream()#Initialize a stream to put all the chords in
    
    for note in song:
        note_idxs = np.where(note[0:128] ==1)[0] #Get all the indices of notes being played
        
        pitch_list = []        
        for p in note_idxs:
            a = music21.pitch.Pitch()
            a.midi = p
            pitch_list.append(a)
        
        chord = music21.chord.Chord(pitch_list)
        dur_idx = np.argmax(note[128:192]) #Get index of duration of note
        chord.duration = music21.duration.Duration(note_duration_frac[dur_idx]) #add duration of chord
        
        stream1.append(chord)
    
    stream_list.append(stream1)
    