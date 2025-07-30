# Utility functions


def get_night(data, target_obsid):
    """
    Given a dictionary mapping obsids to pointings, and the starting obsid,
    return all adjacent obsids for that night and pointing.

    Parameters:
        data (dict): Dictionary of obsid strings mapped to integer values.
        target_obsid (str): The obsid to search around.

    Returns:
        list: A list of adjacent obsids (including the target) from the same night
    """
    sorted_obsids = sorted(data.keys(), key=int)
    target_index = sorted_obsids.index(target_obsid)
    target_value = data[target_obsid]
    result = [target_obsid]

    # Go left
    i = target_index - 1
    while i >= 0 and data[sorted_obsids[i]] == target_value:
        result.insert(0, sorted_obsids[i])
        i -= 1

    # Go right
    i = target_index + 1
    while i < len(sorted_obsids) and data[sorted_obsids[i]] == target_value:
        result.append(sorted_obsids[i])
        i += 1

    return result