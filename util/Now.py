import time


class Now:
    @staticmethod
    def current_timestamp():
        return int(time.time())

    @staticmethod
    def current_dt(date_fmt='%Y%m%d%H%M%S'):
        return time.strftime(date_fmt)

