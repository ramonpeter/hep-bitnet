#!/usr/bin/env python

import os
import glob
import random

# Logging
import logging
logger = logging.getLogger(__name__)

import uproot
import awkward as ak
import numpy as np
import os

def get_chunk( tot, n_split, index):
    ''' Implements split of number into n_split chunks
        https://math.stackexchange.com/questions/2975936/split-a-number-into-n-numbers
        Return tuple (start, stop)
    '''
        
    d = tot // n_split
    r = tot % n_split

    #return [d+1]*r + [d]*(n_split-r)
    if index<r:
        return ( (d+1)*index, (d+1)*(index+1) )
    else:
        return ( (d+1)*r + d*(index-r), (d+1)*r + d*(index-r+1) )

from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):

    def __init__( self, 
            input_files,
            branches            = None, 
            #padding         = 0.,
            n_split             = 1, 
            splitting_strategy  = "files",
            tree_name           = "Events",
            selection           = None,
            max_files           = None,
            verbose             = False,
            shuffle             = False,
                ):
        '''
        DataGenerator for the keras training framework.
        branches: which branches to return 
        n_split:    Number of chunks the data should be split into, use -1 and splitting_stragey = 'files' to split per file.
        input_files:    Input files or directories.
        '''

        # input_files
        self.input_files = []
        for filename in input_files:
            if filename.endswith('.root'):
                self.input_files.extend(glob.glob(filename))
            # step into directory
            elif os.path.isdir( filename ):
                for filename_ in os.listdir( filename ):
                    if filename_.endswith('.root'):
                        self.input_files.append(os.path.join( filename, filename_ ))
            else:
                raise RuntimeError( "Don't know what to do with %r" % filename )

        self.input_files = self.input_files[:max_files]

        random.shuffle( self.input_files )

        self.splitting_strategy = splitting_strategy
        if splitting_strategy.lower() not in ['files', 'events']:
            raise RuntimeError("'splitting_strategy' must be 'files' or 'events'")

        self.verbose = verbose
        self.shuffle = shuffle

        # split per file
        if splitting_strategy == "files" and n_split<0:
            n_split = len(self.input_files)

        # apply selection string
        self.selection = selection        

        # Into how many chunks we split
        self.n_split        = n_split
        if not n_split>0 or not type(n_split)==int:
            raise RuntimeError( "Need to split in positive integer. Got %r"%n )
            
        # variables to return 
        self.branches = branches 

        # recall the index we loaded
        self.index          = None

        # name of the tree to be read
        self.tree_name = tree_name

    def reduceFiles( self, to ):
        self.input_files = self.input_files[:to]
        if self.splitting_strategy == 'files':
            self.n_split = min( [to, self.n_split] )

        print ("Reducing files to %i. n_split: %i" % (len(self.input_files), self.n_split ) ) 

    # interface to Keras
    def __len__( self ):
        return self.n_split

    def _load( self, index, small = None):

        if self.verbose: print ("Loading index %i, strategy: %s"%(index, self.splitting_strategy.lower() ) )
        if index>=0:
            n_split = self.n_split
        else:
            n_split = 1
            index   = 0

        # load the files (don't load the data)
        if self.splitting_strategy.lower() == 'files':
            filestart, filestop = get_chunk( len(self.input_files), n_split, index )
            self.array          = uproot.concatenate([f+':'+self.tree_name for f in self.input_files[filestart:filestop]], self.branches)
            if self.selection is not None:
                len_before = len(self.array)
                self.array = self.array[self.selection(self.array)]
                if self.verbose: print ("Applying selection with efficiency %4.3f" % (len(self.array)/len_before) )
            entry_start, entry_stop = 0, len(self.array)
        elif self.splitting_strategy.lower() == 'events':
            if not hasattr( self, "array" ):
                self.array      = uproot.concatenate([f+':'+self.tree_name for f in self.input_files], self.branches)
                if self.selection is not None:
                    len_before = len(self.array)
                    self.array = self.array[self.selection(self.array)]
                    if self.verbose: print ("Applying selection with efficiency %4.3f" % (len(self.array)/len_before) )
            entry_start, entry_stop = get_chunk( len(self.array), n_split, index )

        if small is not None and small>0:
            entry_stop = min( entry_stop, entry_start+small )

        self.index = index

        if self.shuffle:
            if not hasattr( self, "permutation" ):
                self.permutation = np.random.permutation(range(len(self.array)))

            self.data  = self.array[self.permutation][entry_start:entry_stop]
            return self.data
        else:
            self.data  = self.array[entry_start:entry_stop]
            return self.data

    def __getitem__(self, index):
        if index == self.index:
            return self.data
        else:
            return self._load( index )

    @staticmethod
    def scalar_branches( data, branches ):
        return np.array( [ data[b].to_list() for b in branches ] ).transpose()

    @staticmethod
    def vector_branch( data, branches, padding_target=50, padding_value=0.):
        if type(branches)==str:
            return np.array(ak.fill_none(ak.pad_none( data[branches].to_list(), target=padding_target, clip=True), value=padding_value))
        else:
            return np.array([ np.array(ak.fill_none(ak.pad_none(data[b].to_list(), target=padding_target, clip=True), value=padding_value)).transpose() for b in branches ]).transpose()

if __name__=='__main__':
    import user
    path = os.path.join(user.data_directory, "v6/WZto1L1Nu_HT300")

    data = DataGenerator(
            input_files = [path],
            n_split   = -1,
            splitting_strategy = "files",
            branches  = ["genJet_pt", "genJet_eta"],
            max_files = 10, 
        )
    total = 0
    for i_chunk, chunk in enumerate(data):
        total += len(chunk)
        print ("Loaded chunk %i with %i events."%(i_chunk, len(chunk)))
    print ("Total:", total )

    data = DataGenerator(
            input_files = [path],
            n_split   = 10,
            splitting_strategy = "events",
            branches  = ["genJet_pt", "genJet_eta"],
            max_files = 10, 
        )

    print()

    total = 0
    for i_chunk, chunk in enumerate(data):
        total += len(chunk)
        print ("Loaded chunk %i with %i events."%(i_chunk, len(chunk)))
    print ("Total:", total )
