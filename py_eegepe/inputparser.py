class inputParser():
    def __init__(self):
        self.Results = dict()

    def addParamter(self, name, value):
        self.Results[name] = value

    # @parse.setter
    def parse(self, varin):

        if not isinstance(varin, dict):
            varin = dict(zip(varin[0::2], varin[1::2]))

        for name, value in varin.items():
            self.Results[name] = value


def parse(parser_class, varin):
    parser_class.parse(varin)


def addParamter(parser_class, name, value):
    parser_class.addParamter(name, value)


def inputParserCustom(d, varargin_, d_setting=None, asclass=False):
    v = inputParser()

    if d_setting is not None:
        raise Exception('Place holder setting for checks to pass to addParameter')

    for name, value in d.items():
        addParamter(v, name, value)

        parse(v, varargin_)

    v = v.Results
    d = None

    if asclass:
        v = Dict2Obj(v)

    return v, d


class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """

    # ----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


def obj_to_dict(d):
    """
    Turns a dictionary into a class
    """

    # ----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


if __name__ == 'main':

    # direct use example
    v = inputParser()
    v.addParamter('a', 1)
    v.addParamter('b', 2)
    v.addParamter('c', 3)
    varin = ('a', 5)
    parse(v, varin)

    # through the inputParserCustom
    d = dict()
    d['a'] = 1
    d['b'] = 2
    d['c'] = 3
    arg = ('a', 6)

    v, d = inputParserCustom(d, arg)



