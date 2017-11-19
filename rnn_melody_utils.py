import urllib, os, midi, random, re, string, sys
import numpy as np


debug = ''


# debug = 'overfit'


class MusicDataLoader(object):
    def __init__(self, datadir, config, not_read=False):
        self.datadir = datadir
        self.genedir = None
        self.output_ticks_per_quarter_note = 120
        self.config = config
        self.pointer = {}
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0
        self.negative_data = None
        print('Data loader: datadir: {}'.format(datadir))
        if not not_read:
            self.read_data()

    def read_data(self):
        """
        read_data takes a datadir containing midi files, reads them into training data for an rnn model.
        Midi music information will be a shape aof [2, 0, 3, 0, 5, 0]

        each steps will be fractions of beat notes (32th notes) and each number which is not 0 and 1 is 
        the pitch of the note.
        """

        self.songs = {}
        self.songs['validation'] = []
        self.songs['test'] = []
        self.songs['train'] = []

        files = os.listdir(self.datadir)
        for i, f in enumerate(files):
            song_data = self.read_one_file(self.datadir, f)
            if song_data is None:
                continue
            self.songs['train'].append(song_data)
            print('Read midi %s' % os.path.join(self.datadir, f))

        random.shuffle(self.songs['train'])
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0
        return self.songs

    def read_one_file(self, path, filename):
        try:
            if debug:
                print('Reading {}'.format(os.path.join(path, filename)))
            midi_pattern = midi.read_midifile(os.path.join(path, filename))
        except:
            print ('Error reading {}'.format(os.path.join(path, filename)))
            return None

        song_data = []

        # Tempo:
        ticks_per_quarter_note = midi_pattern.resolution
        if ticks_per_quarter_note % self.output_ticks_per_quarter_note != 0:
            return None
        input_ticks_per_output_tick = ticks_per_quarter_note / self.output_ticks_per_quarter_note

        # Multiply with output_ticks_pr_input_tick for output ticks.
        for track in midi_pattern:
            last_event_input_tick = 0
            not_closed_note = None
            for event in track:
                if len(song_data) >= 3000:
                    return song_data
                if type(event) == midi.events.SetTempoEvent:
                    pass  # These are currently ignored
                elif (type(event) == midi.events.NoteOffEvent) or \
                        (type(event) == midi.events.NoteOnEvent and \
                                     event.velocity == 0):
                    if not_closed_note:
                        if event.data[0] == not_closed_note[0]:
                            event_abs_tick = (event.tick + last_event_input_tick) / input_ticks_per_output_tick
                            pitch = not_closed_note[0]
                            if pitch > self.config.melody_params.pitch_max:
                                pitch = pitch - ((pitch - self.config.melody_params.pitch_max) / 12 + 1) * 12
                            elif pitch < self.config.melody_params.pitch_min:
                                pitch = pitch + ((self.config.melody_params.pitch_min - pitch) / 12 + 1) * 12
                            song_data.append(pitch - self.config.melody_params.pitch_min + 2)
                            for i in range((not_closed_note[1] - event_abs_tick) / 15 - 1):
                                song_data.append(1)
                            song_data.append(0)
                            not_closed_note = None
                elif type(event) == midi.events.NoteOnEvent:
                    begin_tick = (event.tick + last_event_input_tick) / input_ticks_per_output_tick
                    note = event.data[0]
                    if not not_closed_note:
                        not_closed_note = [note, begin_tick]
                last_event_input_tick += event.tick
        return song_data

    def rewind(self, part='train'):
        self.pointer[part] = 0

    def get_batch_rnn(self, batchsize, songlength, part='train'):
        """
          get_batch() returns a batch from self.songs, as a
          pair of tensors song_data with shape [batchsize, songlength].

          Since self.songs was shuffled in read_data(), the batch is
          a random selection without repetition.

        """
        songlength = songlength + 1
        if self.pointer[part] > len(self.songs[part]) - batchsize:
            # return False, [None, None]
            self.pointer[part] = self.pointer[part] % (len(self.songs[part]) - batchsize)
        if self.songs[part]:
            batch = self.songs[part][self.pointer[part]:self.pointer[part] + batchsize]
            self.pointer[part] += batchsize
            batch_songs = np.ndarray(shape=[batchsize, songlength])

            for s in range(len(batch)):
                if len(batch[s]) < songlength:
                    raise 'the length of song is too short'
                begin = random.randint(0, len(batch[s]) - songlength)
                songmatrix = batch[s][begin: begin + songlength]
                batch_songs[s, :] = songmatrix

            return batch_songs[:, 0: songlength - 1], batch_songs[:, 1: songlength]
        else:
            raise 'get_batch() called but self.songs is not initialized.'

    def data_to_song(self, song_name, song_data):
        """
        data_to_song takes a song in internal representation in the shape of
        [song_length] to a midi pattern
        """

        midi_pattern = midi.Pattern([], resolution=int(self.output_ticks_per_quarter_note))
        cur_track = midi.Track([])
        cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=self.config.melody_params.bpm))

        note_not_close = None
        last_note_tick = 0
        for i in range(len(song_data)):
            note = song_data[i]
            if not note_not_close:
                if note > 1:
                    note_not_close = [note, i]
                    event = midi.events.NoteOnEvent(tick=(i - last_note_tick) * 15, velocity=100,
                                                    pitch=note + self.config.melody_params.pitch_min)
                    cur_track.append(event)
            else:
                if note == 0:
                    event = midi.events.NoteOffEvent(tick=(i - note_not_close[1]) * 15, velocity=0,
                                                     pitch=note_not_close[0] + self.config.melody_params.pitch_min)
                    cur_track.append(event)
                    last_note_tick = i
                    note_not_close = None

        if note_not_close:
            event = midi.events.NoteOffEvent(tick=(len(song_data) - note_not_close[1]) * 15, velocity=0,
                                             pitch=note_not_close[0] + self.config.melody_params.pitch_min)
            cur_track.append(event)

        cur_track.append(midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
        midi_pattern.append(cur_track)
        midi.write_midifile(song_name, midi_pattern)

    def store_negative_data(self, negative_data):
        self.negative_data = negative_data

    def seq_train_data(self, batch_size, song_length):
        num_positive_data = random.randint(1, batch_size-1)
        num_negative_data = batch_size - num_positive_data
        positive_datas, _ = self.get_batch_rnn(num_positive_data, song_length)
        positive_labels = [[0, 1]] * num_positive_data
        indices = np.random.randint(0, len(self.negative_data), num_negative_data)
        negative_datas = [self.negative_data[i] for i in indices]
        negative_labels = [[1, 0]] * num_negative_data

        total_datas = np.concatenate([positive_datas, negative_datas], 0)
        total_labels = np.concatenate([positive_labels, negative_labels], 0)
        shuffle_indices = np.random.permutation(np.arange(len(total_labels)))
        return total_datas[shuffle_indices], total_labels[shuffle_indices]

