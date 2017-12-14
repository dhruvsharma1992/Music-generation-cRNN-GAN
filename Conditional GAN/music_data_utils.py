# Tools to load and save midi files for the rnn-gan-project.
# 
# Written by Olof Mogren, http://mogren.one/
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================= 
import urlparse
from urllib2 import urlopen
import os, midi, math, random, re, string, sys
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from io import BytesIO

GENRE      = 0
#COMPOSER   = 1
SONG_DATA  = 1

# INDICES IN BATCHES (LENGTH,FREQ,VELOCITY are repeated self.tones_per_cell times):
TICKS_FROM_PREV_START      = 0
LENGTH     = 1
FREQ       = 2
VELOCITY   = 3

# INDICES IN SONG DATA (NOT YET BATCHED):
BEGIN_TICK = 0

NUM_FEATURES_PER_TONE = 3
NUM_GENRE = 3

debug = ''
#debug = 'overfit'

sources                              = {}
sources['classical']                 = {}
sources['classical']['alkan']        = ['http://www.classicalmidi.co.uk/alkan.htm']
sources['classical']['adam']         = ['http://www.classicalmidi.co.uk/adam.htm']
sources['classical']['aguado']       = ['http://www.classicalmidi.co.uk/aguadodion.htm']


class MusicDataLoader(object):

  def __init__(self, datadir, select_validation_percentage, select_test_percentage, genres, works_per_composer=None, pace_events=False, synthetic=None, tones_per_cell=1, single_composer=None):
    self.datadir = datadir
    self.output_ticks_per_quarter_note = 384.0
    self.tones_per_cell = tones_per_cell
    self.single_composer = single_composer
    self.pointer = {}
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    self.genres = genres
    if synthetic == 'chords':
      self.generate_chords(pace_events=pace_events)
    elif not datadir is None:
      print ('Data loader: datadir: {}'.format(datadir))
      #self.download_midi_data()
      self.read_data(select_validation_percentage, select_test_percentage, works_per_composer, pace_events)

  def download_midi_data(self):
    """
    download_midi_data will download a number of midi files, linked from the html
    pages specified in the sources dict, into datadir. There will be one subdir
    per genre, and within each genre-subdir, there will be a subdir per composer.
    Hence, similar to the structure of the sources dict.
    """
    midi_files = {}
    print("midi data")

    if os.path.exists(os.path.join(self.datadir, 'do-not-redownload.txt')):
      print ( 'Already completely downloaded, delete do-not-redownload.txt to check for files to download.')
      return
    for genre in sources:
      midi_files[genre] = {}
      for composer in sources[genre]:
        midi_files[genre][composer] = []
        for url in sources[genre][composer]:
          print ("url", url)
          try:
              response = urlopen(url)
              data = response.read().decode('latin-1')
          except Exception as r:
              print('error',r)
              continue
          #if 'classicalmidi' in url:
          #  headers = response.info()
          #  print ( headers

          #htmlinks = re.findall('"(  ?[^"]+\.htm)"', data)
          #for link in htmlinks:
          #  print ( 'http://www.classicalmidi.co.uk/'+strip(link)
          
          # make urls absolute:
          urlparsed = urlparse(url)
          
          data = re.sub('href="\/', 'href="http://'+urlparsed.hostname+'/', data, flags= re.IGNORECASE)
          data = re.sub('href="(?!http:)', 'href="http://'+urlparsed.hostname+urlparsed.path[:urlparsed.path.rfind('/')]+'/', data, flags= re.IGNORECASE)
          #if 'classicalmidi' in url:
          #  print ( data
          
          links = re.findall('"(http://[^"]+\.mid)"', data)
          for link in links:
            cont = False
            for p in ignore_patterns:
              if p in link:
                print ( 'Not downloading links with {}'.format(p))
                cont = True
                continue
            if cont: continue
            print ( link)
            filename = link.split('/')[-1]
            valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
            filename = ''.join(c for c in filename if c in valid_chars)
            print ( genre+'/'+composer+'/'+filename)
            midi_files[genre][composer].append(filename)
            localdir = os.path.join(os.path.join(self.datadir, genre), composer)
            localpath = os.path.join(localdir, filename)
            if os.path.exists(localpath):
              print ( 'File exists. Not redownloading: {}'.format(localpath))
            else:
              try:
                response_midi = urlopen(link)
                try: os.makedirs(localdir)
                except: pass
                data_midi = response_midi.read()
                #print(type(data_midi))
                
                #if 'DOCTYPE html PUBLIC' in data_midi:
                #  print ( 'Seems to have been served an html page instead of a midi file. Continuing with next file.')
                #elif 'RIFF' in data_midi[0:9]:
                #  print ( 'Seems to have been served an RIFF file instead of a midi file. Continuing with next file.')
                #else:
                with open(localpath, 'wb') as f:
                    f.write(data_midi)
              except Exception as e:
                print("error 2\n",e)
                #response_midi = urlopen(link)
                #print(response_midi.read())
                print ( 'Failed to fetch {}'.format(link))
    with open(os.path.join(self.datadir, 'do-not-redownload.txt'), 'w') as f:
      f.write('This directory is considered completely downloaded.')

  def generate_chords(self, pace_events):
    """
    generate_chords generates synthetic songs with either major or minor chords
    in a chosen scale.

    returns a list of tuples, [genre, composer, song_data]
    Also saves this list in self.songs.

    Time steps will be fractions of beat notes (32th notes).
    """

    self.genres = ['classical']
    print (('num genres:{}'.format(len(self.genres))))
    self.composers = ['generated_chords']
    print (('num composers: {}'.format(len(self.composers))))

    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []
    
    # https://songwritingandrecordingtips.wordpress.com/2012/02/09/chord-progressions-that-fit-together/
    # M m m M M m
    base_tones     = [0,2,4,5,7,9]
    chord_is_major = [True,False,False,True,True,False]
    #(W-W-H-W-W-W-H)
    #(2 2 1 2 2 2 1)
    major_third_offset = 4
    minor_third_offset = 3
    fifth_offset       = 7

    songlength = 500
    numsongs = 1000

    genre = self.genres[0]
    composer = self.composers[0]
    
    #write_files = False
    #print (('write_files = False')
    #if self.datadir is not None:
    #  write_files = True
    #  print (('write_files = True')
    #  dirnameforallfiles = os.path.join(self.datadir, os.path.join(genre, composer))
    #  if not os.path.exists(dirnameforallfiles):
    #    os.makedirs(dirnameforallfiles)
    #  else:
    #    print (('write_files = False')
    #    write_files = False

    for i in range(numsongs):
      # OVERFIT
      if i % 100 == 99:
        print ( 'Generating songs {}/{}: {}'.format(genre, composer, (i+1)))
      
      song_data = []
      key = random.randint(0,100)
      #key = 50

      # Tempo:
      ticks_per_quarter_note = 384
      
      for j in range(songlength):
        last_event_input_tick=0
        not_closed_notes = []
        begin_tick = float(j*ticks_per_quarter_note)
        velocity = float(100)
        length = ticks_per_quarter_note-1
        
        # randomness out of chords that 'fit'
        # https://songwritingandrecordingtips.wordpress.com/2012/02/09/chord-progressions-that-fit-together/
        base_tone_index = random.randint(0,5)
        base_tone = key+base_tones[base_tone_index]
        is_major = chord_is_major[base_tone_index]
        third = base_tone+major_third_offset
        if not is_major:
          third = base_tone+minor_third_offset
        fifth = base_tone+fifth_offset

        note = [0.0]*(NUM_FEATURES_PER_TONE+1)
        note[LENGTH]     = length
        note[FREQ]       = tone_to_freq(base_tone)
        note[VELOCITY]   = velocity
        note[BEGIN_TICK] = begin_tick
        song_data.append(note)
        note2 = note[:]
        note2[FREQ] = tone_to_freq(third)
        song_data.append(note2)
        note3 = note[:]
        note3[FREQ] = tone_to_freq(fifth)
        song_data.append(note3)
      song_data.sort(key=lambda e: e[BEGIN_TICK])
      #print ((song_data)
      #sys.exit()
      if (pace_events):
        pace_event_list = []
        pace_tick = 0.0
        song_tick_length = song_data[-1][BEGIN_TICK]+song_data[-1][LENGTH]
        while pace_tick < song_tick_length:
          song_data.append([0.0, 440.0, 0.0, pace_tick, 0.0])
          pace_tick += float(ticks_per_quarter_note)/input_ticks_per_output_tick
        song_data.sort(key=lambda e: e[BEGIN_TICK])
      if self.datadir is not None and i==0:
        filename = os.path.join(self.datadir, '{}.mid'.format(i))
        if not os.path.exists(filename):
          print (('saving: {}.'.format(filename)))
          self.save_data(filename, song_data)
        else:
          print (('file exists. Not overwriting: {}.'.format(filename)))
      
      if i%100 == 0:
        self.songs['validation'].append([genre, composer, song_data])
      elif i%100 == 1:
        self.songs['test'].append([genre, composer, song_data])
      else:
        self.songs['train'].append([genre, composer, song_data])
    
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    print (('lens: train: {}, val: {}, test: {}'.format(len(self.songs['train']), len(self.songs['validation']), len(self.songs['test']))))
    return self.songs


  def read_data(self, select_validation_percentage, select_test_percentage, works_per_composer, pace_events):
    """
    read_data takes a datadir with genre subdirs, and composer subsubdirs
    containing midi files, reads them into training data for an rnn-gan model.
    Midi music information will be real-valued frequencies of the
    tones, and intensity taken from the velocity information in
    the midi files.

    returns a list of tuples, [genre, composer, song_data]
    Also saves this list in self.songs.

    Time steps will be fractions of beat notes (32th notes).
    """
    print('num genres:{}'.format(len(self.genres)))
    
    ohe = LabelBinarizer()
    ohe.fit(self.genres)

    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []

    song_data_genre = []

    for genre in self.genres:
        one_hot_genre = ohe.transform([genre])[0]

        current_path = os.path.join(self.datadir, genre)
        if not os.path.exists(current_path):
            print ('Path does not exist: {}'.format(current_path))
            continue

        files = os.listdir(current_path)
        for i,f in enumerate(files):
            print ('Reading files {}/{}: {}'.format(genre, f, (i+1)))
            if os.path.isfile(os.path.join(current_path,f)):
                song_data = self.read_one_file(current_path, f, pace_events)
                if song_data is None:
                    continue
                song_data_genre.append([genre, song_data])

    len_song_data_genre = len(song_data_genre)
    #validation_len = int(float(select_validation_percentage/100.0)*len_song_data_genre
    if select_validation_percentage or select_test_percentage:
        test_len = int(float(select_test_percentage/100.0)*len_song_data_genre)
        validation_len = int(float(select_validation_percentage/100.0)*len_song_data_genre)
        self.songs['test']= song_data_genre[:test_len]
        self.songs['validation']= song_data_genre[test_len:validation_len+test_len]
        self.songs['train']= song_data_genre[validation_len+test_len:]
    else:
        self.songs['train']= song_data_genre
    #print ("Prahal len_song_data_genre", len_song_data_genre)
    # random.shuffle(self.songs['train'])
    print("songs",len(self.songs['train']))
    print("songs",len(self.songs['validation']))
    print("songs",len(self.songs['test']))
    return self.songs




  def read_one_file(self, path, filename, pace_events):
    try:
      if debug:
        print (('Reading {}'.format(os.path.join(path,filename))))
      midi_pattern = midi.read_midifile(os.path.join(path,filename))
    except:
      print ( 'Error reading {}'.format(os.path.join(path,filename)))
      return None
    #
    # Interpreting the midi pattern.
    # A pattern has a list of tracks
    # (midi.Track()).
    # Each track is a list of events:
    #   * midi.events.SetTempoEvent: tick, data([int, int, int])
    #     (The three ints are really three bytes representing one integer.)
    #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
    #     (ignored)
    #   * midi.events.KeySignatureEvent: tick, data([int, int])
    #     (ignored)
    #   * midi.events.MarkerEvent: tick, text, data
    #   * midi.events.PortEvent: tick(int), data
    #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
    #   * midi.events.ProgramChangeEvent: tick, channel, data
    #   * midi.events.ControlChangeEvent: tick, channel, data
    #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
    #
    #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
    #     - data[0] is the note (0-127)
    #     - data[1] is the velocity.
    #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
    #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
    #
    #   * midi.events.EndOfTrackEvent: tick(int), data()
    #
    # Ticks are relative.
    #
    # Tempo are in microseconds/quarter note.
    #
    # This interpretation was done after reading
    # http://electronicmusic.wikia.com/wiki/Velocity
    # http://faydoc.tripod.com/formats/mid.htm
    # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
    # and looking at some files. It will hopefully be enough
    # for the use in this project.
    #
    # We'll save the data intermediately with a dict representing each tone.
    # The dicts we put into a list. Times are microseconds.
    # Keys: 'freq', 'velocity', 'begin-tick', 'tick-length'
    #
    # 'Output ticks resolution' are fixed at a 32th note,
    #   - so 8 ticks per quarter note.
    #
    # This approach means that we do not currently support
    #   tempo change events.
    #
    # TODO 1: Figure out pitch.
    # TODO 2: Figure out different channels and instruments.
    #
    
    song_data = []
    
    # Tempo:
    ticks_per_quarter_note = float(midi_pattern.resolution)
    #print (('Resoluton: {}'.format(ticks_per_quarter_note))
    input_ticks_per_output_tick = ticks_per_quarter_note/self.output_ticks_per_quarter_note
    #if debug == 'overfit': input_ticks_per_output_tick = 1.0
    
    # Multiply with output_ticks_pr_input_tick for output ticks.
    for track in midi_pattern:
      last_event_input_tick=0
      not_closed_notes = []
      for event in track:
        if type(event) == midi.events.SetTempoEvent:
          pass # These are currently ignored
        elif (type(event) == midi.events.NoteOffEvent) or \
             (type(event) == midi.events.NoteOnEvent and \
              event.velocity == 0):
          retained_not_closed_notes = []
          for e in not_closed_notes:
            if tone_to_freq(event.data[0]) == e[FREQ]:
              event_abs_tick = float(event.tick+last_event_input_tick)/input_ticks_per_output_tick
              #current_note['length'] = float(ticks*microseconds_per_tick)
              e[LENGTH] = event_abs_tick-e[BEGIN_TICK]
              song_data.append(e)
            else:
              retained_not_closed_notes.append(e)
          #if len(not_closed_notes) == len(retained_not_closed_notes):
          #  print (('Warning. NoteOffEvent, but len(not_closed_notes)({}) == len(retained_not_closed_notes)({})'.format(len(not_closed_notes), len(retained_not_closed_notes)))
          #  print (('NoteOff: {}'.format(tone_to_freq(event.data[0])))
          #  print (('not closed: {}'.format(not_closed_notes))
          not_closed_notes = retained_not_closed_notes
        elif type(event) == midi.events.NoteOnEvent:
          begin_tick = float(event.tick+last_event_input_tick)/input_ticks_per_output_tick
          note = [0.0]*(NUM_FEATURES_PER_TONE+1)
          note[FREQ]       = tone_to_freq(event.data[0])
          note[VELOCITY]   = float(event.data[1])
          note[BEGIN_TICK] = begin_tick
          not_closed_notes.append(note)
          #not_closed_notes.append([0.0, tone_to_freq(event.data[0]), velocity, begin_tick, event.channel])
        last_event_input_tick += event.tick
      for e in not_closed_notes:
        #print (('Warning: found no NoteOffEvent for this note. Will close it. {}'.format(e))
        e[LENGTH] = float(ticks_per_quarter_note)/input_ticks_per_output_tick
        song_data.append(e)
    song_data.sort(key=lambda e: e[BEGIN_TICK])
    if (pace_events):
      pace_event_list = []
      pace_tick = 0.0
      song_tick_length = song_data[-1][BEGIN_TICK]+song_data[-1][LENGTH]
      while pace_tick < song_tick_length:
        song_data.append([0.0, 440.0, 0.0, pace_tick, 0.0])
        pace_tick += float(ticks_per_quarter_note)/input_ticks_per_output_tick
      song_data.sort(key=lambda e: e[BEGIN_TICK])
    return song_data

  def rewind(self, part='train'):
    self.pointer[part] = 0

  def get_batch(self, batchsize, songlength, part='train'):
    """
      get_batch() returns a batch from self.songs, as a
      pair of tensors (genrecomposer, song_data).
      
      The first tensor is a tensor of genres and composers
        (as two one-hot vectors that are concatenated).
      The second tensor contains song data.
        Song data has dimensions [batchsize, songlength, num_song_features]

      To have the sequence be the primary index is convention in
      tensorflow's rnn api.
      The tensors will have to be split later.
      Songs are currently chopped off after songlength.
      TODO: handle this in a better way.

      Since self.songs was shuffled in read_data(), the batch is
      a random selection without repetition.

      songlength is related to internal sample frequency.
      We fix this to be every 32th notes. # 50 milliseconds.
      This means 8 samples per quarter note.
      There is currently no notion of tempo in the representation.

      composer and genre is concatenated to each event
      in the sequence. There might be more clever ways
      of doing this. It's not reasonable to change composer
      or genre in the middle of a song.
      
      A tone  has a feature telling us the pause before it.

    """
    #print (('get_batch(): pointer: {}, len: {}, batchsize: {}'.format(self.pointer[part], len(self.songs[part]), batchsize))
    #print ('Prahal batch returns ', part, len(self.songs[part]), batchsize ,len(self.songs[part])-batchsize)
    if self.pointer[part] > len(self.songs[part])-batchsize:
      print(self.pointer[part],part,len(self.songs[part]),batchsize)
      # print ('Prahal batch returns 2')
      return [None, None]
    if self.songs[part]:
      batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
      self.pointer[part] += batchsize
      # subtract two for start-time and channel, which we don't include.
      num_meta_features = len(self.genres)#+len(self.composers)
      # All features except timing are multiplied with tones_per_cell (default 1)
      num_song_features = NUM_FEATURES_PER_TONE*self.tones_per_cell+1
      batch_genrecomposer = np.ndarray(shape=[batchsize, num_meta_features])
      batch_songs = np.ndarray(shape=[batchsize, songlength, num_song_features])
      #print ( 'batch shape: {}'.format(batch_songs.shape)
      zeroframe = np.zeros(shape=[num_song_features])
      for s in range(len(batch)):
        songmatrix = np.ndarray(shape=[songlength, num_song_features])
        #composeronehot = onehot(self.composers.index(batch[s][1]), len(self.composers))
        genreonehot = onehot(self.genres.index(batch[s][0]), len(self.genres))
        #genrecomposer = np.concatenate([genreonehot, composeronehot])
        genrecomposer = genreonehot
        
        
        #random position:
        begin = 0
        if len(batch[s][SONG_DATA]) > songlength*self.tones_per_cell:
          begin = random.randint(0, len(batch[s][SONG_DATA])-songlength*self.tones_per_cell)
        matrixrow = 0
        n = begin
        while matrixrow < songlength:
          eventindex = 0
          event = np.zeros(shape=[num_song_features])
          if n < len(batch[s][SONG_DATA]):
            event[LENGTH]   = batch[s][SONG_DATA][n][LENGTH]*1.0/100
            event[FREQ]     = batch[s][SONG_DATA][n][FREQ]*1.0/1000
            event[VELOCITY] = batch[s][SONG_DATA][n][VELOCITY]*1.0/100
            ticks_from_start_of_prev_tone = 0.0
            if n>0:
              # beginning of this tone, minus starting of previous
              ticks_from_start_of_prev_tone = batch[s][SONG_DATA][n][BEGIN_TICK]-batch[s][SONG_DATA][n-1][BEGIN_TICK]
              # we don't include start-time at index 0:
              # and not channel at -1.
            # tones are allowed to overlap. This is indicated with
            # relative time zero in the midi spec.
            event[TICKS_FROM_PREV_START] = ticks_from_start_of_prev_tone *1.0/100

            # event[NUM_FEATURES_PER_TONE-len(self.genres):NUM_FEATURES_PER_TONE] = genreonehot

            # print(genreonehot)
            
            tone_count = 1
            for simultaneous in range(1,self.tones_per_cell):
              if n+simultaneous >= len(batch[s][SONG_DATA]):
                break
              if batch[s][SONG_DATA][n+simultaneous][BEGIN_TICK]-batch[s][SONG_DATA][n][BEGIN_TICK] == 0:
                offset = simultaneous*NUM_FEATURES_PER_TONE
                event[offset+LENGTH]   = batch[s][SONG_DATA][n+simultaneous][LENGTH]
                event[offset+FREQ]     = batch[s][SONG_DATA][n+simultaneous][FREQ]
                event[offset+VELOCITY] = batch[s][SONG_DATA][n+simultaneous][VELOCITY]
                # print(offset+NUM_FEATURES_PER_TONE-len(self.genres),offset+NUM_FEATURES_PER_TONE,genreonehot,event.shape)
                # event[offset+NUM_FEATURES_PER_TONE-len(self.genres):offset+NUM_FEATURES_PER_TONE] = genreonehot
                tone_count += 1
              else:
                break
          songmatrix[matrixrow,:] = event
          matrixrow += 1
          n += tone_count
        #if s == 0 and self.pointer[part] == batchsize:
        #  print ( songmatrix[0:10,:]
        batch_genrecomposer[s,:] = genrecomposer
        batch_songs[s,:,:] = songmatrix
      #batched_sequence = np.split(batch_songs, indices_or_sections=songlength, axis=1)
      #return [np.squeeze(s, axis=1) for s in batched_sequence]
      #print (('batch returns [0:10]: {}'.format(batch_songs[0,0:10,:]))
      #print ('Prahal batch returns 3 ', len(batch_genrecomposer), batch_genrecomposer[0])
      # print("get_batch",batch_genrecomposer, batch_songs)
      return batch_genrecomposer, batch_songs
    else:
      raise 'get_batch() called but self.songs is not initialized.'
  
  def get_num_song_features(self):
    return NUM_FEATURES_PER_TONE*self.tones_per_cell+1
  def get_num_meta_features(self):
    #print ("Prahal ", len(self.genres))
    return len(self.genres)#+len(self.composers)

  def get_midi_pattern(self, song_data):
    """
    get_midi_pattern takes a song in internal representation 
    (a tensor of dimensions [songlength, self.num_song_features]).
    the three values are length, frequency, velocity.
    if velocity of a frame is zero, no midi event will be
    triggered at that frame.

    returns the midi_pattern.

    Can be used with filename == None. Then nothing is saved, but only returned.
    """
    #print (('song_data[0:10]: {}'.format(song_data[0:10])))


    #
    # Interpreting the midi pattern.
    # A pattern has a list of tracks
    # (midi.Track()).
    # Each track is a list of events:
    #   * midi.events.SetTempoEvent: tick, data([int, int, int])
    #     (The three ints are really three bytes representing one integer.)
    #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
    #     (ignored)
    #   * midi.events.KeySignatureEvent: tick, data([int, int])
    #     (ignored)
    #   * midi.events.MarkerEvent: tick, text, data
    #   * midi.events.PortEvent: tick(int), data
    #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
    #   * midi.events.ProgramChangeEvent: tick, channel, data
    #   * midi.events.ControlChangeEvent: tick, channel, data
    #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
    #
    #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
    #     - data[0] is the note (0-127)
    #     - data[1] is the velocity.
    #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
    #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
    #
    #   * midi.events.EndOfTrackEvent: tick(int), data()
    #
    # Ticks are relative.
    #
    # Tempo are in microseconds/quarter note.
    #
    # This interpretation was done after reading
    # http://electronicmusic.wikia.com/wiki/Velocity
    # http://faydoc.tripod.com/formats/mid.htm
    # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
    # and looking at some files. It will hopefully be enough
    # for the use in this project.
    #
    # This approach means that we do not currently support
    #   tempo change events.
    #
    
    # Tempo:
    # Multiply with output_ticks_pr_input_tick for output ticks.
    midi_pattern = midi.Pattern([], resolution=int(self.output_ticks_per_quarter_note))
    cur_track = midi.Track([])
    cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=45))
    future_events = {}
    last_event_tick = 0
    
    ticks_to_this_tone = 0.0
    song_events_absolute_ticks = []
    abs_tick_note_beginning = 0.0
    for frame in song_data:
      abs_tick_note_beginning += int(frame[TICKS_FROM_PREV_START]*100)
      for subframe in range(self.tones_per_cell):
        offset = subframe*NUM_FEATURES_PER_TONE
        tick_len           = abs(int(round(frame[offset+LENGTH]*100)))
        freq               = frame[offset+FREQ]*1000
        velocity           = min(int(round(frame[offset+VELOCITY]*100)),127)
        #print (('tick_len: {}, freq: {}, velocity: {}, ticks_from_prev_start: {}'.format(tick_len, freq, velocity, frame[TICKS_FROM_PREV_START]))
        d = freq_to_tone(freq)
        #print (('d: {}'.format(d))
        if d is not None and velocity > 0 and tick_len > 0:
          # range-check with preserved tone, changed one octave:
          tone = d['tone']
          while tone < 0:   tone += 12
          while tone > 127: tone -= 12
          pitch_wheel = cents_to_pitchwheel_units(d['cents'])
          #print (('tick_len: {}, freq: {}, tone: {}, pitch_wheel: {}, velocity: {}'.format(tick_len, freq, tone, pitch_wheel, velocity))
          #if pitch_wheel != 0:
          #midi.events.PitchWheelEvent(tick=int(ticks_to_this_tone),
          #                                            pitch=pitch_wheel)
          song_events_absolute_ticks.append((abs_tick_note_beginning,
                                             midi.events.NoteOnEvent(
                                                   tick=0,
                                                   velocity=velocity,
                                                   pitch=tone)))
          song_events_absolute_ticks.append((abs_tick_note_beginning+tick_len,
                                             midi.events.NoteOffEvent(
                                                    tick=0,
                                                    velocity=0,
                                                    pitch=tone)))
    song_events_absolute_ticks.sort(key=lambda e: e[0])
    abs_tick_note_beginning = 0.0
    for abs_tick,event in song_events_absolute_ticks:
      rel_tick = abs_tick-abs_tick_note_beginning
      event.tick = int(round(rel_tick))
      cur_track.append(event)
      abs_tick_note_beginning=abs_tick
    
    cur_track.append(midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
    midi_pattern.append(cur_track)
    #print ( 'print (ing midi track.'
    #print ( midi_pattern
    return midi_pattern

  def save_midi_pattern(self, filename, midi_pattern):
    if filename is not None:
      midi.write_midifile(filename, midi_pattern)

  def save_data(self, filename, song_data):
    """
    save_data takes a filename and a song in internal representation 
    (a tensor of dimensions [songlength, 3]).
    the three values are length, frequency, velocity.
    if velocity of a frame is zero, no midi event will be
    triggered at that frame.

    returns the midi_pattern.

    Can be used with filename == None. Then nothing is saved, but only returned.
    """
    midi_pattern = self.get_midi_pattern(song_data)
    self.save_midi_pattern(filename, midi_pattern)
    return midi_pattern

def tone_to_freq(tone):
  """
    returns the frequency of a tone. 

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0

def freq_to_tone(freq):
  """
    returns a dict d where
    d['tone'] is the base tone in midi standard
    d['cents'] is the cents to make the tone into the exact-ish frequency provided.
               multiply this with 8192 to get the midi pitch level.

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  if freq <= 0.0:
    return None
  float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
  int_tone = int(float_tone)
  cents = int(1200*math.log(float(freq)/tone_to_freq(int_tone), 2))
  return {'tone': int_tone, 'cents': cents}

def cents_to_pitchwheel_units(cents):
  return int(40.96*(float(cents)))

def onehot(i, length):
  a = np.zeros(shape=[length])
  a[i] = 1.0
  return a



def main():
  filename = sys.argv[1]
  print (('File: {}'.format(filename)))
  dl = MusicDataLoader(datadir=None, select_validation_percentage=0.0, select_test_percentage=0.0)
  print (('length, frequency, velocity, time from previous start.'))
  abs_song_data = dl.read_one_file(os.path.dirname(filename), os.path.basename(filename), pace_events=True)
  
  rel_song_data = []
  last_start = None
  for i,e in enumerate(abs_song_data):
    this_start = e[3]
    if last_start:
      e[3] = e[3]-last_start
    rel_song_data.append(e)
    last_start = this_start
    print ((e))
  if len(sys.argv) > 2:
    if not os.path.exists(sys.argv[2]):
      print (('Saving: {}.'.format(sys.argv[2])))
      dl.save_data(sys.argv[2], rel_song_data)
    else:
      print (('File already exists: {}. Not saving.'.format(sys.argv[2])))
if __name__ == "__main__":
  main()

