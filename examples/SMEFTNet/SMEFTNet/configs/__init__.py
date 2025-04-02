#import os
#__path__.append(os.path.dirname(os.path.abspath(__file__)))
import sys, os
_path = os.path.dirname(os.path.abspath(__file__))
#print (_path)
sys.path.insert( 0, _path)
sys.path.insert( 0,'/'.join(_path.split('/')[:-1]))
