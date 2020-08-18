from torch import zeros


class CutAndPaste:
    def __init__(self, max_frames):
        self.max_frames = max_frames

    def __call__(self, wave):
        length = len(wave)
        if length % 2 == 1:
            wave = wave[:-1]
        if length > self.max_frames:
            mid_idx = length // 2
            wave = wave[mid_idx - self.max_frames // 2:mid_idx + self.max_frames // 2]
        elif length < self.max_frames:
            mid_idx = self.max_frames // 2
            t = zeros(self.max_frames)
            t[mid_idx - length // 2:mid_idx + length // 2] = t[mid_idx - length // 2:mid_idx + length // 2] + wave
            wave = t
        return wave
