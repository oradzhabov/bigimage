import inspect
import logging


def json_def_encoder(obj):
    if inspect.isclass(obj):
        if hasattr(obj, 'json_class_encode'):
            return obj.json_class_encode()
        logging.error('Trying to serialize class which have no required interface')
        return dict()
    if hasattr(obj, 'json_instance_encode'):
        return obj.json_instance_encode()
    return obj.__dict__


class JSONEncoder(object):

    @classmethod
    def json_class_encode(cls):
        result = dict()
        result['class'] = cls.__name__
        return result

    def json_instance_encode(self):
        result = dict()
        result['class'] = type(self).__name__
        return result
