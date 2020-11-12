# singapore-music-classifier

<!-- test -->


## Features

### Low-level features
Low-level features are extracted using the opensource audio feature extranction tool, [OpenSMILE](https://www.audeering.com/opensmile/) 

| Feature Name | Definition |
| --- | --- |
| F0final | The smoothed fundamental frequency contour |
| voicingFinalUnclipped | The voicing probability of the final fundamental frequency candidate. Unclipped means,that it was not set to zero when is falls below the voicing threshold|
| jitterLocal | The local (frame-to-frame) Jitter (pitch period length deviations) |
| jitterDDP | The differential frame-to-frame Jitter (the ‘Jitter of the Jitter’) |
| shimmerLocal | The local (frame-to-frame) Shimmer (amplitude deviations between pitch periods) |
| logHNR | Log of the ratio of the energy of harmonic signal components to the energy of noise like signal components |
| audspec_lengthL1norm | Magnitude of L1 norm of Auditory Spectrum |
| audspecRasta_lengthL1norm | Relative Spectral Transform applied to Auditory Spectrum and lengthL1norm is the magnitude of the L1 norm |
| pcm_RMSenergy | Root-mean-square signal frame energy |
| pcm_zcr | Zero-crossing rate of time signal (frame-based) |
| audSpec_Rfilt (0 ~ 25) | Relative Spectral Transform (RASTA)-style filtered applied to Auditory Spectrum |
| pcm_fftMag_fband250-650 | fft magnitude of frequency band between 250Hz to 650Hz |
| pcm_fftMag_fband1000-4000 | fft magnitude of frequency band between 1000Hz to 4000Hz |
| pcm_fftMag_psySharpness | Psychoacoustic sharpness |
| pcm_fftMag_spectralHarmonicity | Spectral Harmonicity |
| pcm_fftMag_mfcc(1 ~ 14) | Mel-frequency cepstral coefficients 1–14 |
| pcm_fftMag_spectralRollOff25.0 | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralRollOff50.0 | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralRollOff75.0 | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralRollOff90.0 | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralFlux | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralCentroid | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralEntropy | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralVariance | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralSkewness | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralKurtosis | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |
pcm_fftMag_spectralSlope | Spectral Features (frequency based features) which are obtained by converting time-based signal into frequency domain using the Fourier Transform |


### High-level features

High-level features are extracted by using [Essentia](https://essentia.upf.edu/algorithms_reference.html) Python interface

| Descriptor type | Feature name | Property | Algorithm | Definition |
| --- | --- | --- | --- | --- |
| Rhythm  |  beats per minute (bpm) | | Rhythm Extractor2013 | beats per minute - the tempo estimation |
| Rhythm  |  ticks | Temporal | Rhythm Extractor2013 | the estimated tick locations |
| Rhythm  |  bpmintervals (Beats per minute intervals) | Temporal | Rhythm Extractor2013 | list of beats intervals |
| Rhythm  |  loudness | Temporal | Beats Loudness | the beat´s energy in the whole spectrum |
| Rhythm  |  danceability | | Danceability | the danceability value. Normal values range from 0 to 3. The higher, the more danceable |
| Rhythm  |  Detrended Fluctuation Analysis (DFA) | | Danceability | the Detrended Fluctuation Analysis (DFA) exponent vector for considered segment length (tau) values |
| Tonal | key | | KeyExtractor | the estimated key, from A to G |
| Tonal | scale | | KeyExtractor | the scale of the key (major or minor) | 
| Tonal | strength | | Strength | the strength of the estimated key | 
| Tonal | chords_changes_rate | | TonalExtractor | the rate at which chords change in the progression | 
| Tonal | chords_histogram | Temporal | TonalExtractor | the normalized histogram of chords | 
| Tonal | chords_key | | TonalExtractor | the most frequent chord of the progression | 
| Tonal | chords_number_rate | | TonalExtractor | the ratio of different chords from the total number of chords in the progression| 
| Tonal | chords_scale | | TonalExtractor | the scale of the most frequent chord of the progression (either ’major’ or ’minor’) | 
| Tonal | chords_strength | Temporal | TonalExtractor | the strength of the chord | 
| Tonal | key_key | | TonalExtractor | the estimated key, from A to G | 
| Tonal | key_scale | | TonalExtractor | the scale of the key (major or minor) | 
| Tonal | key_strength | | TonalExtractor | the strength of the estimated key | 
| Tonal | tuningFrequency | Temporal | Tuning-Frequency-Extractor | the computed tuning frequency | 
| Spectral | frequencies | Temporal | SpectralPeaks | the frequencies of the spectral peaks [Hz] |
| Spectral | magnitude | Temporal | SpectralPeaks | the magnitudes of the spectral peaks |
| Pitch | salienceFunction | Temporal | PitchSalience-Function | array of the quantized pitch salience values |
| Pitch | salienceBins | Temporal | PitchSalience-FunctionPeaks |This algorithm computes the peaks of a given pitch salience function |
| Pitch | salienceValues | Temporal | PitchSalience-FunctionPeaks | salience values corresponding to the peaks |
| Loudness | dynamicComplexity | | DynamicComplexity| the dynamic complexity coefficient |
| Loudness | estimate-loudness | | DynamicComplexity | an estimate of the loudness [dB] |
| Loudness | intensity | | Intensity | the intensity value |
| Loudness | larm | | Larm | This algorithm estimates the long-term loudness of an audio signal. |
| Loudness | leq | | Leq | This algorithm computes the Equivalent sound level (Leq) of an audio signal |
| Loudness | loudness (loud algorithm) | Temporal | LevelExtractor | This algorithm extracts the loudness of an audio signal in frames using Loudness algorithm|
| Loudness | loudness (Steven power algorithm)| | Loudness | This algorithm computes the loudness of an audio signal defined by Steven’s power law |
