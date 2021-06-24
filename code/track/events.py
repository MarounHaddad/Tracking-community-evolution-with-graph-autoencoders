"""
This file contains the list of functions that are used to calculate the events of the sequences
We consider the following events:
    MISSING = 0
    FORM = 1
    CONTINUE = 2
    GROW = 3
    SHRINK = 4
    SPLIT = 5
    MERGE = 6
    DISSOLVE = 7
    REFORM = 8
"""

# Libraries
from enum import Enum
from tabulate import tabulate
import datapreparation.preprocess as pp


class event_type(Enum):
    MISSING = 0
    FORM = 1
    CONTINUE = 2
    GROW = 3
    SHRINK = 4
    SPLIT = 5
    MERGE = 6
    DISSOLVE = 7
    REFORM = 8


sequence_formed = False


def get_events(sequences):
    """
    Calculates the events of the sequences
    :param sequences: the list of generated sequences
    :return: the list of events per sequence per snapshot(timestep)
    """
    events = []
    global sequence_formed
    for sequence in sequences:
        events.append([])

    all_sequences_print = []
    for sequence_index in range(0, len(sequences)):
        sequence_print = (sequence_index,)
        sequence_formed = False
        for time_step in range(0, pp.number_timesteps):
            events[sequence_index].append(get_event(sequences[sequence_index], time_step))
            str_event = str(events[sequence_index][time_step]).replace("event_type.", "")
            sequence_print = sequence_print + (str_event,)
        all_sequences_print.append(sequence_print)

    header_print = ["sequence"]
    for time_step in range(0, pp.number_timesteps):
        header_print.append(time_step)

    print(tabulate(all_sequences_print, headers=header_print))
    return events

def total_size(clusters):
    size = 0
    for cluster in clusters:
        size += pp.clusters_lookup[cluster][2]
    return size


def is_missing(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) == 0 and len(clusters_current_timestep) == 0:
        return True
    else:
        return False


def is_form(clusters_current_timestep, clusters_previous_timestep):
    global sequence_formed
    if len(clusters_previous_timestep) == 0 and len(clusters_current_timestep) != 0:
        sequence_formed = True
        return True
    else:
        return False


def is_dissolve(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) != 0 and len(clusters_current_timestep) == 0:
        return True
    else:
        return False


def is_split(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) < len(clusters_current_timestep) and \
            len(clusters_current_timestep) != 0 and len(clusters_previous_timestep) != 0:
        return True
    else:
        return False


def is_merge(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) > len(clusters_current_timestep) and \
            len(clusters_current_timestep) != 0 and len(clusters_previous_timestep) != 0:
        return True
    else:
        return False


def is_continue(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) == len(clusters_current_timestep) and \
            total_size(clusters_current_timestep) == total_size(clusters_previous_timestep):
        return True
    else:
        return False


def is_grow(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) == len(clusters_current_timestep) and \
            total_size(clusters_current_timestep) > total_size(clusters_previous_timestep):
        return True
    else:
        return False


def is_shrink(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) == len(clusters_current_timestep) and \
            total_size(clusters_current_timestep) < total_size(clusters_previous_timestep):
        return True
    else:
        return False


def is_reform(clusters_current_timestep, clusters_previous_timestep):
    if len(clusters_previous_timestep) == 0 and len(clusters_current_timestep) != 0 and sequence_formed:
        return True
    else:
        return False


def get_event(sequence, timestep):
    """
    This function calculates the events per sequence per timestep
    :param sequence: the sequence of whom we want to calculate the event
    :param timestep: The snapshot ID
    :return:
    """
    clusters_current_timestep = pp.get_sequence_clusters_in_timestep(sequence, timestep)
    if timestep == 0:
        clusters_previous_timestep = []
    else:
        clusters_previous_timestep = pp.get_sequence_clusters_in_timestep(sequence, timestep - 1)

    if is_missing(clusters_current_timestep, clusters_previous_timestep):
        return event_type.MISSING
    elif is_dissolve(clusters_current_timestep, clusters_previous_timestep):
        return event_type.DISSOLVE
    elif is_reform(clusters_current_timestep, clusters_previous_timestep):
        return event_type.REFORM
    elif is_form(clusters_current_timestep, clusters_previous_timestep):
        return event_type.FORM
    elif is_split(clusters_current_timestep, clusters_previous_timestep):
        return event_type.SPLIT
    elif is_merge(clusters_current_timestep, clusters_previous_timestep):
        return event_type.MERGE
    elif is_grow(clusters_current_timestep, clusters_previous_timestep):
        return event_type.GROW
    elif is_shrink(clusters_current_timestep, clusters_previous_timestep):
        return event_type.SHRINK
    elif is_continue(clusters_current_timestep, clusters_previous_timestep):
        return event_type.CONTINUE
    else:
        return -1

