import audiomentations


def get_transforms_train_wave():
    wave_transforms = audiomentations.Compose([
                            audiomentations.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=0.5),
                            audiomentations.TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
                            audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                            audiomentations.Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
                            ])
    return wave_transforms


def get_transforms_train_wave_custom(custom):
    wave_transforms = audiomentations.Compose(custom)
    return wave_transforms