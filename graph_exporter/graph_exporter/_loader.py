import yaml


class IgnoreUnknownTagsLoader(yaml.SafeLoader):
    def construct_undefined(self, node):
        return None


IgnoreUnknownTagsLoader.add_constructor(
    None, IgnoreUnknownTagsLoader.construct_undefined
)
