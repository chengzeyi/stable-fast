import io
import itertools
import pstats
import prettytable


class ProfileParser:

    def __init__(self, *amount):
        self.amount = amount

    def __call__(self, pr):
        stats_stream = io.StringIO()
        stats = pstats.Stats(pr, stream=stats_stream).sort_stats(
            pstats.SortKey.CUMULATIVE)
        caller_time = stats.total_tt
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if len(callers) == 0 and not (func[:2] == ('~', 0)):
                stats.top_level.add(func)
        width, list = stats.get_print_list(self.amount)
        top_level = stats.top_level
        return ProfileParseResult(top_level, caller_time, list,
                                  [stats.stats[func] for func in list])


TABULAR_FIELD_NAMES = [
    'Caller',
    'Caller time',
    'Number of calls',
    'Total time',
    'Cumulative time',
    'Callee',
]


class ProfileParseResult:

    def __init__(self, top_level, caller_time, funcs, stats):
        self.top_level = top_level
        self.caller_time = caller_time
        self.funcs = funcs
        self.stats = stats

    def get_tablular(self):
        table = prettytable.PrettyTable()
        table.field_names = TABULAR_FIELD_NAMES
        for caller, total_time, nc, tt, ct, callee in itertools.zip_longest(
            [pstats.func_get_function_name(func) for func in self.top_level],
            [self.caller_time],
            [stat[1] for stat in self.stats],
            [stat[2] for stat in self.stats],
            [stat[3] for stat in self.stats],
            [
                pstats.func_std_string(pstats.func_strip_path(func))
                for func in self.funcs
            ],
                fillvalue='',
        ):
            table.add_row([caller, total_time, nc, tt, ct, callee])
        return table


class ProfileParseResults:

    def __init__(self, results=None):
        self.results = []
        if results is not None:
            for result in results:
                self.add(result)

    def add(self, result):
        self.results.append(result)

    def clear(self):
        self.results.clear()

    def get_tablular(self):
        table = prettytable.PrettyTable()
        table.field_names = TABULAR_FIELD_NAMES
        for i, result in enumerate(self.results):
            sub_table = result.get_tablular()
            for row in sub_table.rows:
                table.add_row(row)
            if i < len(self.results) - 1:
                table.add_row(['*'] * len(TABULAR_FIELD_NAMES))
        return table
