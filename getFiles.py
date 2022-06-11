p='https://pmm-gv.gsfc.nasa.gov/pub/gpm-validation/data/gpmgv/netcdf/geo_match/GPM/2BDPRGMI/V06A/2_0/2021/'
p='https://pmm-gv.gsfc.nasa.gov/pub/gpm-validation/data/gpmgv/netcdf/geo_match/GPM/2BDPRGMI/V06A/2_0/2018/'

import os
#import urllib2
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Fetch the html file

import datetime

sd=datetime.datetime(2022,5,1)

for iday in range(30):
    cd=sd+datetime.timedelta(days=iday)
    dd=cd.day
    mm=cd.month
    try:
        p='https://pmm-gv.gsfc.nasa.gov/pub/gpm-validation/data/gpmgv/orbit_subset/GPM/DPR/2ADPR/V07A/KWAJ/2022/%2.2i/%2.2i/'%(mm,dd)
        p='https://pmm-gv.gsfc.nasa.gov/pub/gpm-validation/data/gpmgv/orbit_subset/GPM/DPR/2ADPR/V07A/CONUS/2022/%2.2i/%2.2i/'%(mm,dd)
        response = urlopen(p)
        html_doc = response.read()
        soup = BeautifulSoup(html_doc, 'html.parser')
        lines=soup.select('a')
        for l in lines:
            if "HDF5" in str(l):
                #print(l)
                l=str(l)
                fname=l[9:-4]
                print(fname)
                #stop
                cmd='wget -nc '+p+'%s'%fname[:73]
                print(cmd)
                os.system(cmd)
    except:
        continue
#for f in fs[1:]:
#    f1=f.split()[2]
#    cmd='wget  '+p+'%s'%f1
#    if ('1801' in f1) or ('1802' in f1) or ('1803' in f1):
#        print(cmd)
#        os.system(cmd)
#    #stop
    #
