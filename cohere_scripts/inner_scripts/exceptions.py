class CohereUiException(Exception):
    """

    """
class CohereUiNotSupported(CohereUiException):
    """
    The same behavior as in standard exception with notion it is raised by cohere_ui

    Attributes:
        message -- explanation of the error
    """

class CohereUiMissingConfParam(CohereUiException):
    """
    The same behavior as in standard exception with notion it is raised by cohere_ui

    Attributes:
        message -- explanation of the error
    """

class CohereUiMissingFileDir(CohereUiException):
    """
    The same behavior as in standard exception with notion it is raised by cohere_ui

    Attributes:
        message -- explanation of the error
    """
