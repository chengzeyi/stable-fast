import sys
from .pretty_profile import (ProfileParser, ProfileParseResults)


class AutoProfiler:

    def __init__(self, *amount, file=None):
        if file is None:
            file = sys.stdout
        self.file = file
        self.amount = amount
        self.parser = ProfileParser(None, *amount)
        self.results = ProfileParseResults()

    def with_cProfile(self, func):
        from .cprofile import with_cProfile
        return with_cProfile(*self.amount,
                             out_func=self.out_func)(func)

    def out_func(self, pr):
        result = self.parser(pr)
        self.results.add(result)

    def __enter__(self):
        self.results.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.print_tablular()

    def print_tablular(self):
        table = self.results.get_tablular().copy()
        table.float_format = '.3'
        table.max_width = 80
        print(table, file=self.file)
