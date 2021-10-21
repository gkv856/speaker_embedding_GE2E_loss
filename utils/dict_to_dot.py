class DictWithDotNotation(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DictWithDotNotation(value)
            self[key] = value


class GetDictWithDotNotation(DictWithDotNotation):

    def __init__(self, hp_dict):
        super(DictWithDotNotation, self).__init__()

        hp_dotdict = DictWithDotNotation(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = DictWithDotNotation.__getitem__
    __setattr__ = DictWithDotNotation.__setitem__
    __delattr__ = DictWithDotNotation.__delitem__


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    test_dict = {

        "key1": 1,
        "key2": 2,
    }
    hp = GetDictWithDotNotation(test_dict)
    print(hp.key1)

