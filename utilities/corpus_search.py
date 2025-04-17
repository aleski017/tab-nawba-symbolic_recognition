from utilities.constants import *
import json
import pandas as pd


def load_pd(pd_path):
    ''' Load a pitch distribution from a json file and return the values (vals and bins)

    :param pd_path: path of the json file
    :return: vals and related bins of the pitch distribution
    '''
    pd = json.load(open(pd_path))
    vals = pd["vals"]
    bins = pd["bins"]
    return vals, bins


def get_id_track_by_nawba(nawba_id = None):
    with open(DF_PATH + "andalusian_description.json") as file:
        data = (json.load(file))
    track_list = []
    for entry in data:
        mbid = entry.get('mbid')
        for section in entry.get('sections', []):
            if section['nawba']['id'] == nawba_id:
                track_list.append(mbid)
                break
    return track_list


def get_id_label(label):
    """
    Parameter
        label: target label to retrieve all instances. E.g. all tab in the df
    --------------------------------------------------------------------------

    Returns a list of all target labels in the dataset
    """
    with open(DF_PATH + "andalusian_description.json") as file:
        data = (json.load(file))
    track_labels = []
    for entry in data:
        for section in entry.get('sections', []):
            label_track = section[label]['id']
            if label_track:
                track_labels.append(label_track)
                break
    return track_labels


def get_all_id_tracks():
    with open(DF_PATH + "andalusian_description.json") as file:
        data = (json.load(file))
    track_list = []
    for entry in data:
        mbid = entry.get('mbid')
        track_list.append(mbid)
    return track_list


def get_ids_tracks_with_multiple_label(label):
    """
    Parameters
        label: either mizan, tab or nawba
    --------------------------------------------------------------------------------

    Returns a list of of track ids that have multiple unique labels for each track.

    Each id should have only one nawba and tab label
    """

    with open(DF_PATH + "andalusian_description.json") as file:
            data = (json.load(file))
    track_list = []
    for entry in data:
        mbid = entry.get('mbid')
        prev_label = 0
        for section in entry.get('sections', []):
            if section[label]['id'] != prev_label and prev_label != 0:
                track_list.append(mbid)
                break
            prev_label = section[label]['id']
    return track_list

with open("corpus-dataset/andalusian_description.json") as file:
    data = (json.load(file))


def parse_all_tracks_sections():
    """
    Parses the 'andalusian_description.json' and returns a dataframe with the following colomns:
    ['start_time', 'tab', 'nawba', 'end_time', 'form', 'mizan', 'track_id']
    """
    section_list = []
    mbids = []
    for entry in data:
        for section in entry.get('sections', []):
            mbids.append(entry.get('mbid'))
            section_list.append(section)

    df_section = pd.DataFrame(section_list)
    df_section['track_id'] = mbids

    return df_section

