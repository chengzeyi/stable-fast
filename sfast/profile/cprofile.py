import io
import functools
import cProfile
import pstats


def with_cProfile(*amount, out_func=None, file=None):

    def _with_cProfile(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            try:
                retval = pr.runcall(func, *args, **kwargs)
                return retval
            finally:
                if out_func is None:
                    stats_stream = io.StringIO()
                    stats = pstats.Stats(pr, stream=stats_stream).sort_stats(
                        pstats.SortKey.CUMULATIVE)
                    stats.print_stats(*amount)
                    msg = stats_stream.getvalue()
                    if file is None:
                        print(msg)
                    else:
                        print(msg, file=file)
                else:
                    out_func(pr)

        return wrapper

    return _with_cProfile
