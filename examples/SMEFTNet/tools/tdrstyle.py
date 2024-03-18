import ROOT
import os
ROOT.gROOT.LoadMacro(os.path.join(os.path.dirname(__file__), "scripts/tdrstyle.C"))
ROOT.setTDRStyle()
