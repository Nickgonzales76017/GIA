import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Softmax
from kapre import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer
#import argparse
#import sys
import codecs
#from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import os
from scipy.io import wavfile
import soundfile as sf
import numpy as np
from tensorflow.python.keras.backend import zeros
import tensorflow_io as tfio
from pydub import AudioSegment


#sys.path.append("../AutoencoderJuce/external/frugally-deep/keras_export/")



# creates a txt document called ListAnno
#and fills it with a list of dataset resorces

directory = './dataset2/annotation/'
F = open("./dataset2/ListAnno.txt", "w")
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        #print(f)
        f = f + '\n'
        F.write(f)
F.close()
# parses a txt document called ListAnno
#and pulls the events out to format for nueral net
ddirectory ="./dataset2/audio/"
for filename1 in os.listdir(ddirectory):
  #print(os.listdir(ddirectory))
  test ="./test/vas.wav"
  sound = AudioSegment.from_wav(test)
  sound = sound.set_channels(1)
  sound.export(test, format="wav")
  if '.wav' in filename1:
    jack = (filename1.replace('.wav',''))

  Song = ddirectory + filename1

  Sheetmusic = "./dataset2/annotation/" + jack + ".xml"
  data, none = sf.read(Song)
  samplerate = 44000
  sf.write(Song, data, samplerate, subtype='PCM_16')
  [samplerate, data] = wavfile.read(Song)
  datat, nonet = sf.read(test)
  sampleratet = 44000
  sf.write(test, datat, sampleratet, subtype='PCM_16')
  [sampleratet, datat] = wavfile.read(test)
  datat = np.array(datat[0:2244000]) 
  #print(len(datat))
  newarr = np.array_split(datat, len(datat)/44000)


  root = ET.parse(Sheetmusic).getroot()
  variables = {}
  counter = 0
  for event in root.iter('event'):
    #print(counter, event.tag)
    counter += 1
    name = []
    value = []
    for child in event:
      name.append(child.tag)
      value.append(child.text)
    variables[counter] = {tuple(zip(name, value))}
  #print('Enter number of event of interest:')
  #x = input()
  ONSET = []
  PITCH = []
  OFFSET = []
  EXCSTYLE = []
  EXPSTYLE = []
  FRETNUMBER = []
  STRINGNUM = []
  MODFREQ = []
  DSA = []
  time =[]
  for EVENT in range(1, counter): 
    ONSET.append(np.array(str(list(variables[EVENT])[0][0][1]),dtype=object))
    PITCH.append(np.array(str(list(variables[EVENT])[0][1][1]),dtype=object))
    OFFSET.append(np.array(str(list(variables[EVENT])[0][2][1]),dtype=object))
    EXCSTYLE.append(np.array(str(list(variables[EVENT])[0][3][1]),dtype=object))
    EXPSTYLE.append(str(list(variables[EVENT])[0][4][1]))
    FRETNUMBER.append(np.array(str(list(variables[EVENT])[0][5][1]),dtype=object))
    STRINGNUM.append(np.array(str(list(variables[EVENT])[0][6][1]),dtype=object))
    MODFREQ.append(np.array(str(list(variables[EVENT])[0][7][1]),dtype=object))
    onset_time = float(ONSET[EVENT-1])* samplerate
    offset_time = float(OFFSET[EVENT-1])* samplerate
    tim = samplerate
    offset_time1 =  onset_time + tim
    eventAudio = np.array(data[round(onset_time):round(offset_time1)],dtype=float)
    sf.write("./AR_A_fret_0-201.wav", eventAudio, samplerate, subtype='PCM_16')
    [samplerate1, data1] = wavfile.read("./AR_A_fret_0-201.wav")
    #print (len(data1))
    DSA.append(data1)
    time.append(offset_time-onset_time)
  #print(time)
  #print(DSA)    


  dataset1 = tf.data.Dataset.from_tensor_slices(EXPSTYLE)
  dataset2 = tf.data.Dataset.from_tensor_slices(DSA)

  #0 onset
  #1 pitch
  #2 offset
  #3 excitation style
  #4 expression Style
  #5 fret number
  #6 string number
  #7 modulation freq
  #print(list(variables[EVENT])[0][2][1])

  max_time = data.size/samplerate
  eventAudio = np.asarray(data[round(onset_time):round(offset_time)]).astype(np.float)
  #print(eventAudio)



  #EX1 =['PK','MU','FS']# 3
  #EX2 =np.array(['BE','DN','FL','HA','NO','SL','ST','TR','VI']) #9
  Batch = counter-1 #EVENT.size
  #print(EX2)
  # 1 channels (!), maybe 1-sec audio signal, for an example.
  input_shape = (samplerate, 1)
  sr = samplerate
  model = Sequential()
  # A STFT layer
  model.add(STFT(n_fft=2048, win_length=2018, hop_length=1024,
                window_name=None, pad_end=False,
                input_data_format='channels_last', output_data_format='channels_last',
                input_shape=input_shape))
  model.add(Magnitude())
  model.add(MagnitudeToDecibel())  # these three layers can be replaced with get_stft_magnitude_layer()
  # Alternatively, you may want to use a melspectrogram layer
  # melgram_layer = get_melspectrogram_layer()
  # or log-frequency layer
  # log_stft_layer = get_log_frequency_spectrogram_layer() 

  # add more layers as you want
  model.add(Conv2D(32, (3, 3), strides=(2, 2)))
  model.add(BatchNormalization())
  model.add(ReLU())
  model.add(GlobalAveragePooling2D())
  model.add(Dense(9))
  model.add(Softmax())

  # Compile the model
  model.compile('adam', 'categorical_crossentropy') # if single-label classification
  def show_shapes(Sequences,Targets): # can make yours to take inputs; this'll use local variable values
      print("Expected: (num_samples, timesteps, channels)")
      print("Sequences: {}".format(Sequences.shape))
      print("Targets:   {}".format(Targets.shape)) 
  # train it with raw audio sample inputs
  # for example, you may have functions that load your data as below.

  x =(np.array(DSA).reshape( Batch, samplerate, 1).astype(np.float))
  #print(x)
  cnt = -1
  #['BE','DN','FL','HA','SL','ST','TR','VI','NO']
  for I in EXPSTYLE:
    cnt += 1
    if I == 'NO':
      EXPSTYLE[cnt] = [0,0,0,0,0,0,0,0,1]
    elif I == 'BE':
      EXPSTYLE[cnt] = [1,0,0,0,0,0,0,0,0]
    elif I == 'DN':
      EXPSTYLE[cnt] = [0,1,0,0,0,0,0,0,0]
    elif I == 'FL':
      EXPSTYLE[cnt] = [0,0,1,0,0,0,0,0,0]
    elif I == 'HA':
      EXPSTYLE[cnt] = [0,0,0,1,0,0,0,0,0]
    elif I == 'SL':
      EXPSTYLE[cnt] = [0,0,0,0,1,0,0,0,0]
    elif I == 'ST':
      EXPSTYLE[cnt] = [0,0,0,0,0,1,0,0,0]
    elif I == 'TR':
      EXPSTYLE[cnt] = [0,0,0,0,0,0,1,0,0]
    elif I == 'VI':
      EXPSTYLE[cnt] = [0,0,0,0,0,0,0,1,0]
    else:
      continue
  y = (np.array(EXPSTYLE).reshape( Batch, 9).astype(int)) # e.g., y.shape = (Batch, 10) if it's 10-class classification
  #print (y)
  #show_shapes(x,y)
  # then..
  model.fit(x,y,epochs=10)
o = model.predict(np.array(newarr).astype(float))
for to in o:
  print(to)
  #oO = o.index(max(np.array(o).astype(float)))
  oO = np.where(to == max(np.array(to).astype(float))).astype(int)
  if oO == 8:
    put =  'NO'
  elif oO == 0:
    put = 'BE'
  elif oO == 1:
    put = 'DN'
  elif oO == 2:
    put = 'FL'
  elif oO == 3:
    put = 'HA'
  elif oO == 4:
    put = 'SL'
  elif oO == 5:
    put = 'ST'
  elif oO == 6:
    put = 'TR'
  elif oO == 7:
    put = 'VI'
  print(oO)
  # Done!

