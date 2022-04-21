import os


class FileExtFilter():
    def __init__(self, ext=['tif', 'jpg']):
        self.ext = ext

    def filter(self, filename):
        file_ext = filename.split('.')[-1]
        return file_ext in self.ext