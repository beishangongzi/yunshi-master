import logging


class Logger:
    @staticmethod
    def get_logger(name='main', log_file='log.txt'):
        handler = logging.FileHandler(filename=log_file)
        # handler.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        # console.setLevel(logging.INFO)

        logger = logging.getLogger(name)
        logger.setLevel(level=logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger

    @staticmethod
    def get_file_logger(name='main', log_file='log.txt'):
        handler = logging.FileHandler(filename=log_file)
        logger = logging.getLogger(name)
        logger.setLevel(level=logging.DEBUG)
        logger.addHandler(handler)
        return logger
