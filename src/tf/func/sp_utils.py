import numpy as np
import jenkspy as jnb


def get_sp_uncertain(uncertain_matrix, sp_map, exclude_sp=None):
    """
    get superpixel uncertainty
    Args:
        uncertain_matrix: pixel uncertainty
        sp_map: superpixel map, integer labels

    Returns: uncertainty value list, one value for each superpixel which is the mean of all included pixel uncertainty

    """
    score_list = []
    for i in np.unique(sp_map):
        if exclude_sp is not None:
            if i in exclude_sp:
                continue
        score_list.append(np.sum(uncertain_matrix[sp_map == i]) / np.sum(sp_map == i))
    return score_list


def get_keep_sp_cluster(sp_uncertain, numbuer_class=2):
    """
    cluster superpixels into keep and query by uncertainty
    Args:
        sp_uncertain: superpixel uncertainty value list, in increasing label order

    Returns: list of superpixel label values to keep

    """
    breaks = jnb.jenks_breaks(sp_uncertain, n_classes=numbuer_class)
    # (lower bound 0, higher bound 0, higher bound 1)
    break_point = breaks[-2]
    keep_list = np.argwhere(np.array(sp_uncertain) <= break_point)
    return np.squeeze(keep_list, axis=-1)


def get_keep_sp_percentage(sp_uncertain, keep_percent):
    """
    return lower uncertainty superpixels by percentage
    Args:
        sp_uncertain: superpixel uncertainty value list, in increasing label order

    Returns: list of superpixel label values to keep

    """
    index_sorted = np.argsort(sp_uncertain)
    return index_sorted[:int(len(sp_uncertain) * keep_percent)]


def get_sp_label(label_map, sp_map, sp_val):
    """
    take a pixelwise label map, get superpixel label value
    each superpixel can have only one label
    superpixels that have multiple pixel label classes take majority label

    """
    sub_map = label_map[sp_map == sp_val]
    sub_value, sub_count = np.unique(sub_map, return_counts=True)
    return sub_value[np.argmax(sub_count)]


def merge_sp_mask(manual_label, prediction, uncertain_matrix, sp_map, gen_percent=None, use_ignore=False,
                  use_clustering=True, sp_keep_percent=None, number_cluster=2,
                  class_pixel_count=None, class_sp_count=None):
    new_mask = np.zeros(manual_label.shape)
    sp_uncertain = get_sp_uncertain(uncertain_matrix, sp_map)
    if use_clustering:
        sp_keep = get_keep_sp_cluster(sp_uncertain, numbuer_class=number_cluster)
    else:
        sp_keep = get_keep_sp_percentage(sp_uncertain, sp_keep_percent)
    sp_query = list(set(np.unique(sp_map)) - set(sp_keep))
    query_mask = np.zeros(manual_label.shape)

    for i in sp_keep:
        # write keep masks
        if use_ignore:
            new_mask[sp_map == i] = 255
        else:
            # new_mask[sp_map == i] = get_sp_label(prediction, sp_map, i)
            new_mask[sp_map == i] = prediction[sp_map == i]
        query_mask[sp_map == i] = 1

    for i in sp_query:
        # new_mask[sp_map == i] = get_sp_label(manual_label, sp_map, i)
        sp_lbl = manual_label[sp_map == i]
        new_mask[sp_map == i] = sp_lbl
        query_mask[sp_map == i] = 0
        if class_sp_count is not None:
            class_present = np.unique(sp_lbl)
            for c in class_present:
                if c != 255:
                    class_sp_count[c] += 1

    if gen_percent is not None:
        # track number of blank sps and total sps.
        gen_percent['gen_sp'].append(len(sp_keep))
        gen_percent['total_sp'].append(len(sp_keep) + len(sp_query))

    if class_pixel_count is not None:
        # track labeled pixels by class
        class_pixel, cp_count = np.unique(new_mask, return_counts=True)
        for i, c in enumerate(class_pixel.astype(np.uint8)):
            if c != 255:
                class_pixel_count[c] += cp_count[i]

    return new_mask, query_mask


def merge_fg_sp_mask(manual_label, prediction, uncertain_matrix, sp_map, gen_percent=None, use_ignore=False,
                     use_clustering=True, sp_keep_percent=None, number_cluster=2):
    new_mask = np.zeros(manual_label.shape)
    fg_sp, bg_sp = get_fg_bg_sp(prediction, sp_map)
    sp_uncertain = get_sp_uncertain(uncertain_matrix, sp_map, exclude_sp=bg_sp)
    if use_clustering:
        sp_keep_i = get_keep_sp_cluster(sp_uncertain, numbuer_class=number_cluster)
    else:
        sp_keep_i = get_keep_sp_percentage(sp_uncertain, sp_keep_percent)
    sp_keep = [fg_sp[x] for x in sp_keep_i]
    sp_query = list(set(fg_sp) - set(sp_keep))
    query_mask = np.ones(manual_label.shape)

    for i in sp_keep:
        # write keep masks
        if use_ignore:
            new_mask[sp_map == i] = 255
        else:
            # new_mask[sp_map == i] = get_sp_label(prediction, sp_map, i)
            new_mask[sp_map == i] = prediction[sp_map == i]
        query_mask[sp_map == i] = 1

    for i in sp_query:
        # new_mask[sp_map == i] = get_sp_label(manual_label, sp_map, i)
        sp_lbl = manual_label[sp_map == i]
        new_mask[sp_map == i] = sp_lbl
        query_mask[sp_map == i] = 0

    new_mask[prediction == 0] = 0
    if gen_percent is not None:
        # track number of blank sps and total sps.
        gen_percent['gen_sp'].append(len(sp_keep))
        gen_percent['fg_sp'].append(len(sp_keep) + len(sp_query))
        gen_percent['total_sp'].append(len(fg_sp)+len(bg_sp))
    return new_mask, query_mask


def get_non_bg_image_uncertain(pixel_uncertain, pred_map, include_first_class=False):
    """
    non-bg pixels
    """
    if np.all(pred_map == 0):
        return 0
    if include_first_class:
        return np.mean(pixel_uncertain)
    else:
        # penalize all bgs
        non_bg_uncertainty = np.mean(pixel_uncertain[np.squeeze(pred_map, axis=-1) != 0])
        bg_penalty = np.sum(pred_map != 0) / pred_map.numpy().size
        return non_bg_uncertainty * bg_penalty


def get_fg_bg_sp(pred, sp_map):
    """
    Get list of superpixels that are not entirely bg and entirely bg
    """
    fg_sp = []
    bg_sp = []
    for sp in np.unique(sp_map):
        if np.sum(pred[sp_map == sp]) != 0:
            fg_sp.append(sp)
        else:
            bg_sp.append(sp)
    return fg_sp, bg_sp
