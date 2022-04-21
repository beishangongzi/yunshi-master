def get_map_val(map:dict, key, defaultVal):
    if key in map.keys():
        return map[key]
    else:
        return defaultVal
