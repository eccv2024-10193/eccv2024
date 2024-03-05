import copy


def object_to_dict(obj) -> dict:
    """
    Convert an object into a dictionary.

    Private attributes (i.e. begins with __) are ignored.
    """
    d = copy.deepcopy(obj.__dict__)
    for k, v in d.items():
        # Ignore private attributes.
        if k.startswith("__"):
            continue

        # # If enum, convert into string.
        if hasattr(v, "name") and hasattr(v, "value"):
            d[k] = v.value
        # If object, recursively convert into dictionary.
        elif hasattr(v, "__dict__"):
            d[k] = object_to_dict(v)

    return d