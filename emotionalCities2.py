# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:09:29 2024

@author: alice
"""

import mne.time_frequency

import mne
from mne.preprocessing import ICA, corrmap, create_eog_epochs 
from mne.viz import plot_topomap
from mne.viz import plot_ica_scores
from mne.preprocessing import ICA
from autoreject import AutoReject

import asrpy

import matplotlib.pyplot as plt

import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

 
# Carregar o ficheiro EGI
sample_data_raw_file = ("C:\\Users\\alice\\OneDrive\\Ambiente de Trabalho\\NSNP807p_mff.mff")
raw = mne.io.read_raw_egi(sample_data_raw_file, preload=True)
 
#raw.crop(tmin=0, tmax=3).load_data() #Corta os dados se necessário

#Plot posição dos elétrodos no plano e em 3D
# raw.plot_sensors(show_names=True)
# raw.plot_sensors(kind="3d", ch_type="all")

################################# FILTRO ####################################

raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)

#raw.plot(remove_dc=False) #, duration=...

############################ BAD CHANNELS ###################################

raw.info['bads'] = ['E9', 'E45', 'E53', 'E81', 'E132', 'E186']  #Marca os canais maus
raw.info

############################## CRIAR NOVAS ANOTAÇÕES ###############################

#raw.annotations.save("C:\\Users\\alice\\OneDrive\\Ambiente de Trabalho\\saved-annotations1.txt", overwrite=True)
annot_from_file = mne.read_annotations("C:\\Users\\alice\\OneDrive\\Ambiente de Trabalho\\NSNP807.txt")
print(annot_from_file)

raw.set_annotations(annot_from_file)
raw.plot(remove_dc=False) #, duration=...

# raw.annotations.save("C:\\Users\\alice\\OneDrive\\Ambiente de Trabalho\\NSNP807.txt", overwrite=True)


################################### EOG #####################################

# eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name = ['E31', 'E37', 'E18'], reject_by_annotation= True)
# eog_epochs.plot_image(combine="mean")
# eog_epochs.average().plot_joint()

# print(len(eog_epochs))  # Quantidade de épocas de EOG detectadas

eog_evoked = create_eog_epochs(raw, ch_name = ['E31', 'E37', 'E18']).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()
eog_evoked.plot_image()


################################### ICA #####################################

#raw.crop(tmax=60.0).pick(picks=["eeg", "stim"])
#raw.load_data()

filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

ica = ICA(n_components=20, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica

explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)

ica.plot_components(inst=raw)

# blinks
ica.plot_overlay(raw, exclude=[0], picks="eeg")

#Explorar propriedades de cada componente, pôr componente no parâmetro picks
#ica.plot_properties(raw, picks=[5])

ica.exclude = [2, 3, 7]  # indices chosen based on various plots above

# pick some channels that clearly show heartbeats and blinks
regexp = r"(E.)"
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_raw.plot(
    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
)
del reconst_raw

#---

ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['E31', 'E37', 'E18'])
ica.exclude = eog_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(raw, show_scrollbars=False)

# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)

# clean up memory before moving on
# del raw, ica

################################### EVOKED/EPOCHS #####################################

events = mne.find_events(raw, stim_channel="STI 014")
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=23)

print(epochs)

event_dict = {
    "Imagem 1s": 10,
    "Início do vídeo": 11,
    "Pergunta Valência": 12,
    "Resposta Valência": 13,
    "Pergunta Arousal": 14,
    "Resposta Arousal": 15,
    "Ignorar": 9,
}

epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=23, event_id=event_dict, preload=True)
print(epochs.event_id)
del raw  # we're done with raw, free up some memory

epochs.plot(n_epochs=1, events=True)

epochs.plot_drop_log()
print(epochs.drop_log)







































################################# PSD #####################################
#Gráfico Psd 
raw.plot_psd(fmin=0, fmax=40) #Método Welch
raw.plot(remove_dc=False) #, duration=...

#Gráfico
raw.compute_psd(fmax=40).plot(picks="data", exclude="bads")

raw.compute_psd(fmax=50).plot()

################################# TOPO_PLOT ##################################

spectrum = raw.compute_psd(fmin=0, fmax=40)
spectrum.plot(average=True, picks="data", exclude="bads")

























spectrum = raw.compute_psd(fmax=40)
spectrum.plot(average=True, picks="data", exclude="bads", amplitude=False)

# raw.crop(0, 20).load_data() 

# filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

# events = mne.find_events(raw, stim_channel="STI 014")
# print(events[:5])  # show the first 5

# event_dict = {
#     "visual": 9,
# }

# fig = mne.viz.plot_events(
#     events, event_id=event_dict, sfreq=raw.info["sfreq"], first_samp=raw.first_samp
# )

# reject_criteria = dict(
#     eeg=150e-6,  # 150 µV pelo Teorema de Nyquist
# )

########################### ESTIMULOS #################################
# testing_data_folder = mne.datasets.testing.data_path()
# eeglab_raw_file = testing_data_folder / "EEGLAB" / "test_raw.set"
# eeglab_raw = mne.io.read_raw_eeglab(eeglab_raw_file)
# print(eeglab_raw.annotations)

# print(len(eeglab_raw.annotations))
# print(set(eeglab_raw.annotations.duration))
# print(set(eeglab_raw.annotations.description))
# print(eeglab_raw.annotations.onset[0])

# events_from_annot, event_dict = mne.events_from_annotations(eeglab_raw)
# print(event_dict)
# print(events_from_annot[:5])

# custom_mapping = {"rt": 77, "square": 42}
# (events_from_annot, event_dict) = mne.events_from_annotations(
#     eeglab_raw, event_id=custom_mapping
# )
# print(event_dict)
# print(events_from_annot[:5])

# mapping = {
#     1: "auditory/left",
#     2: "auditory/right",
#     3: "visual/left",
#     4: "visual/right",
#     5: "smiley",
#     32: "buttonpress",
# }
# annot_from_events = mne.annotations_from_events(
#     events=events,
#     event_desc=mapping,
#     sfreq=raw.info["sfreq"],
#     orig_time=raw.info["meas_date"],
# )
# raw.set_annotations(annot_from_events)

# raw.plot(start=5, duration=5)



# info = mne.io.read_info(sample_data_raw_file)
# print(info)

# print(info["nchan"])
# eeg_indices = mne.pick_types(info, meg=False, eeg=True)
# print(mne.pick_info(info, eeg_indices)["nchan"])

# print(raw)


# sampling_freq = raw.info["sfreq"]
# start_stop_seconds = np.array([11, 13]) #Extrai uma fração de EGG dos 11 aos 13 s
# start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
# channel_index = 0
# raw_selection = raw[channel_index, start_sample:stop_sample]
# print(raw_selection)

# x = raw_selection[1]
# y = raw_selection[0].T
# plt.plot(x, y)

# channel_names = ["E1"]
# two_meg_chans = raw[channel_names, start_sample:stop_sample]
# y_offset = np.array([5e-11, 0])  # just enough to separate the channel traces
# x = two_meg_chans[1]
# y = two_meg_chans[0].T + y_offset
# lines = plt.plot(x, y)
# plt.legend(lines, channel_names)

