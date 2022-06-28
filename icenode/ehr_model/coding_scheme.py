from ..utils import OOPError


class AbstractCodingScheme:

    @classmethod
    def dx_output(cls):
        raise OOPError('Should be overriden')

    @classmethod
    def dx_input(cls):
        raise OOPError('Should be overriden')

