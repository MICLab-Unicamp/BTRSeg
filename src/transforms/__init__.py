'''
This package contais a collection of transforms to data.
'''
import time
import logging


class Compose():
    '''
    Executes all transforms given in tranlist (as a list)
    '''
    def __init__(self, tranlist, time_trans=False):
        self.tranlist = tranlist
        self.time_trans = time_trans

    def addto(self, tran, begin=False, end=False):
        assert begin != end, "either add to begin or end"
        if begin:
            self.tranlist = [tran] + self.tranlist
        elif end:
            self.tranlist = self.tranlist + [tran]

    def __call__(self, img, mask):
        for tran in self.tranlist:
            begin = time.time()

            img, mask = tran(img, mask)

            logging.debug("{} took {}s".format(tran, time.time() - begin))

        logging.debug("-------- Composed Transforms Finished ---------")

        return img, mask

    def __str__(self):
        string = ""
        for tran in self.tranlist:
            string += str(tran) + ' '
        return string[:-1]


class Transform():
    def __call__(self, img, tgt):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
