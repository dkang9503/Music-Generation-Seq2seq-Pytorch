# Music-Generation-Seq2seq-Pytorch
Seq2seq model to generate sequences of music

Code is very sloppy, so I'll outline the general idea of each file

music21_helper.py is file of functions written to convert every midi file into a list of chords (notes were all compressed into a chord)
loading_classical_data.py is file where a list of MIDI files in a folder were all converted into multi-hot tensors of chords 
encoderdecoder.py is the file of the actual model with pytorch. Manipulates data for proper input and trains the model with 400 epochs
convert_tbpredicted.py is file used to change songs that are going to be used to feed in to the predict model into pytorch tensors
prediction.py is file used to load the model from training and then use it to predict a series of 200 notes given an input sequence.

classical_notes.pkl is a result of loading_classical_data.py
