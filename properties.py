class MelodyParam:
    def __init__(self, ticks_max=240, ticks_min=0, length_max=240, length_min=15, pitch_max=84, pitch_min=40,
                 velocity_max=127, velocity_min=100):
        self.ticks_max = ticks_max
        self.ticks_min = ticks_min
        self.length_max = length_max
        self.length_min = length_min
        self.pitch_max = pitch_max
        self.pitch_min = pitch_min
        self.velocity_max = velocity_max
        self.velocity_min = velocity_min
        self.bpm = 45

        self.nor_ticks = (ticks_max - ticks_min) // 15 + 1
        self.nor_length = (length_max - length_min) // 15 + 1
        self.nor_pitch = pitch_max - pitch_min + 1
        self.nor_velocity = velocity_max - velocity_min

        self.total_length = self.nor_ticks + self.nor_length + self.nor_pitch + self.nor_velocity

        pitch_rate = 1
        velocity_rate = 0.25

        self.ticks_weight = 1
        self.length_weight = 1
        self.pitch_weight = 1
        self.velocity_weight = 0.2