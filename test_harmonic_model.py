import tensorflow as tf
import numpy as np
import glob
import os.path
import soundfile as sf
import matplotlib.pyplot as plt
import tsms


def lr_scheduler():
    def scheduler(epoch, lr):
        if (epoch + 1) % 100 == 0:
            return max(lr * 0.9, 1e-3)
        return lr

    return tf.keras.callbacks.LearningRateScheduler(scheduler)


def main():
    file_list = glob.glob("samples/*.wav")
    index = 0
    audio_file = file_list[index]
    note_number = int(audio_file[-11:-8])

    audio, sample_rate = sf.read(audio_file)

    audio = tf.cast(audio, dtype=tf.float32)
    audio = tf.reshape(audio, shape=(1, -1))

    frame_step = 64

    f0_estimate = tsms.core.midi_to_f0_estimate(
        note_number, audio.shape[1], frame_step)

    refined_f0_estimate, _, _ = tsms.core.refine_f0(
        audio, f0_estimate, sample_rate, frame_step)

    h_freq, h_mag, h_phase = tsms.core.iterative_harmonic_analysis(
        signals=audio,
        f0_estimate=refined_f0_estimate,
        sample_rate=sample_rate,
        frame_step=frame_step,
        db_threshold=-180.0,
        semitone_variation_tol=1.0)

    print(h_freq.shape)

    harmonic_model = tsms.sound_models.HarmonicModel(
        sample_rate=sample_rate,
        frame_step=frame_step,
        channels=h_freq.shape[0],
        frames=h_freq.shape[1],
        harmonics=h_freq.shape[2],
        h_freq=h_freq, h_mag=h_mag, h_phase=h_phase,
        generate_phase=False)

    # harmonic model identification using optimization
    # epochs = 500
    #
    # harmonic_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #     loss=tsms.sound_models.ResidualError(),
    #     run_eagerly=False)
    #
    # harmonic_model.fit(audio, audio,
    #                    epochs=epochs,
    #                    callbacks=[lr_scheduler()])

    harmonic = harmonic_model([])
    harmonic = harmonic[:, :audio.shape[1]]

    residual = audio - harmonic

    harmonic_model.generate_phase = True
    no_phase = harmonic_model([])

    original = np.squeeze(audio.numpy())
    harmonic = np.squeeze(harmonic.numpy())
    residual = np.squeeze(residual.numpy())
    no_phase = np.squeeze(no_phase.numpy())

    filename = os.path.basename(audio_file)
    sf.write(f'samples/outs/{filename}_original.wav', original, sample_rate)
    sf.write(f'samples/outs/{filename}_harmonic.wav', harmonic, sample_rate)
    sf.write(f'samples/outs/{filename}_residual.wav', residual, sample_rate)
    sf.write(f'samples/outs/{filename}_no_phase.wav', no_phase, sample_rate)

    # plot results

    def specgrams(x, title):
        plt.figure(figsize=(6.5, 7))
        plt.subplot(2, 1, 1)
        plt.specgram(x, NFFT=256, Fs=sample_rate, window=None,
                     noverlap=256 - frame_step, mode='psd', vmin=-180)
        plt.title(title + ' spectrogram - fft_size = 256')
        plt.subplot(2, 1, 2)
        plt.specgram(x, NFFT=1024, Fs=sample_rate, window=None,
                     noverlap=1024 - frame_step, mode='psd', vmin=-180)
        plt.title(title + ' spectrogram - fft_size = 1024')

    f0_harmonic = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)

    plt.figure()
    plt.plot(np.squeeze(f0_estimate.numpy()), label='f0 midi')
    plt.plot(np.squeeze(refined_f0_estimate.numpy()), label='f0 correlation')
    plt.plot(np.squeeze(f0_harmonic.numpy()), label='f0 harmonic')
    plt.legend()
    plt.draw()

    plt.figure()
    plt.plot(original, label='original')
    plt.plot(harmonic, label='harmonic')
    plt.plot(residual, label='residual')
    plt.legend()
    plt.draw()

    specgrams(original, title='original')
    specgrams(harmonic, title='harmonic')
    specgrams(residual, title='residual')

    plt.figure()
    h_freq = harmonic_model.h_freq
    plt.plot(np.squeeze(h_freq.numpy()))
    plt.title('frequency of sinusoidal tracks')

    plt.show()


if __name__ == '__main__':
    main()
