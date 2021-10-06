__author__ = 'jamesoneill2'

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getfilenames  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load result file and create result api object.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".


import json
import datetime
import numpy as np

class Flickr:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = []
        self.cats = []

        if not annotation_file is None:
            print ('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            print (datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):

        # create index
        print('creating index...')
        imgToAnns = {}
        for ann in self.dataset['images']:
            for sent in ann['sentences']:
                imgToAnns[sent['imgid']] = [] #sent["raw"]
                imgToAnns[ann['filename']] = []

        anns = {}
        for ann in self.dataset['images']:
            for sent in ann['sentences']:
                anns[sent['sentid']] = [] # sent["raw"]

        """
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']] += [ann]
            anns[ann['id']] = ann

        imgs      = {im['id']: {} for im in self.dataset['images']}
        for img in self.dataset['images']:
            imgs[img['id']] = img                
        """

        for ann in self.dataset['images']:
            for sent in ann['sentences']:
                imgToAnns[sent['imgid']] += [sent['raw']]
                anns[sent['sentid']] = sent

        imgs = {}
        for ann in self.dataset['images']:
            for sent in ann['sentences']:
                imgs[sent['imgid']] = sent # sent["raw"]
                imgs[sent['imgid']].update({'filename': ann['filename'], 'split': ann['split']})

        cats = []
        catToImgs = []
        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnIds(self, filenames=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param filenames  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        filenames = filenames if type(filenames) == list else [filenames]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(filenames) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(filenames) == 0:
                anns = sum([self.imgToAnns[filename] for filename in filenames if filename in self.imgToAnns],[])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

        ids = [ann['sentids'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['sentids']            in catIds]
        ids = [cat['sentids'] for cat in cats]
        return ids

    def getfilenames(self, filenames=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param filenames (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        filenames = filenames if type(filenames) == list else [filenames]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(filenames) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(filenames)
            for catId in catIds:
                if len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            for sent in ann['sentences']:
                print(sent['raw'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = Flickr()
        res.dataset['images'] = [img for img in self.dataset['images']]
        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns  = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsfilenames = [ann['filename'] for ann in anns]
        assert set(annsfilenames) == (set(annsfilenames) & set(self.getfilenames())), \
            'Results do not correspond to current coco set'
        filenames = set([img['sentids'] for img in res.dataset['images']]) & set([ann['filename'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['sentids'] in filenames]
        for id, ann in enumerate(anns):
            ann['sentids'] = id
        res.dataset['annotations'] = anns
        res.createIndex()
        return res


    @staticmethod
    def decodeMask(R):
        """
        Decode binary mask M encoded via run-length encoding.
        :param   R (object RLE)    : run-length encoding of binary mask
        :return: M (bool 2D array) : decoded binary mask
        """
        N = len(R['counts'])
        M = np.zeros( (R['size'][0]*R['size'][1], ))
        n = 0
        val = 1
        for pos in range(N):
            val = not val
            for c in range(R['counts'][pos]):
                R['counts'][pos]
                M[n] = val
                n += 1
        return M.reshape((R['size']), order='F')

    @staticmethod
    def encodeMask(M):
        """
        Encode binary mask M using run-length encoding.
        :param   M (bool 2D array)  : binary mask to encode
        :return: R (object RLE)     : run-length encoding of binary mask
        """
        [h, w] = M.shape
        M = M.flatten(order='F')
        N = len(M)
        counts_list = []
        pos = 0
        # counts
        counts_list.append(1)
        diffs = np.logical_xor(M[0:N-1], M[1:N])
        for diff in diffs:
            if diff:
                pos +=1
                counts_list.append(1)
            else:
                counts_list[pos] += 1
        # if array starts from 1. start with 0 counts for 0
        if M[0] == 1:
            counts_list = [0] + counts_list
        return {'size':      [h, w],
               'counts':    counts_list ,
               }
