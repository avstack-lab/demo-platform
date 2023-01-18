class AppendPipeline():
    """A dummy pipeline that appends a string and sends back"""
    def __init__(self, append_string) -> None:
        self.append_string = append_string.encode("ascii")

    def __call__(self, in_string: str) -> str:
        return in_string + self.append_string


class AnalyzeVideo():
    """Analyzes videos with tracks"""


def init_pipeline(pipeline_type):
    if 'append' in pipeline_type:
        return AppendPipeline(pipeline_type.replace('append-', ''))
    else:
        raise NotImplementedError(pipeline_type)
