import os, sys
import numpy as np
import pandas as pd
# only meant for reading spr_file's
# spr_files have following structure
# 1. magic keyword (optional) 
#    .spr (SPRAAK package) or .key (HMM package)  otherwise NOHEADER file is assumed
# 2.  header (optional)
#   a header with multiple lines consisting of  KEY VALUE pairs where
#            the first word is the KEY and the REMAINDER the VALUE
# 3. separator (required if 1. or 2. or present)
#    line starting with "#"
# 4. data 
#   formatted as specified in the header or via other arguments
#
#   noheader files are assumed to be text and read as sequence of lines
#     e.g. dictionaries, phone lists, ...
#
def spr_open_file(fname,flag="r",encoding="latin1",noheader=False):
    if flag != "r":
        print("open_spr_file(): is designed for READING spr_files only")
        exit
    if not os.path.isfile(fname):
       print("File path {} does not exist. Exiting...".format(fname))
       sys.exit()
    fp = open(fname,'r',encoding=encoding)
    first_time = True
    hdr = {}
        

    while(1):
        line = fp.readline()
        line = line.strip()
        # determine the header type of the file
        if ( first_time ):
            first_time = False
            if( line == ".key" ) : hdr['HEADER'] = "key"
            elif (line == ".spr") : hdr['HEADER'] = "spr"
            else:
                hdr['HEADER'] = "nohdr"
                hdr['DATA'] = "LIST"
                fp.seek(0)
#                break
            continue
        # continue reading header KEY VALUE pairs till EOH is detected
        if len(line) == 0: continue
        elif line[0]=="#": 
            break
        else:
            w = line.split(None,1)
            if len(w) == 1: hdr[w[0]] = True
            else: hdr[w[0]] = w[1]
        
    return fp, hdr

def spr_read_data(fp,hdr):
    if 'DATATYPE' in hdr.keys():
        hdr['DATA'] = hdr['DATATYPE']
        
    if not 'DATA' in hdr.keys(): hdr['DATA'] = 'LIST' 

    if(hdr['DATA']=='TRACK'):
        nfr = int(hdr['DIM1'])
        nparam = int(hdr['DIM2'])
        itemtype = 'float32'
#        print(nfr,nparam,itemtype)
        data = np.fromfile(fp,dtype=itemtype,count=-1)
        data = np.reshape(data,(nfr,nparam))
    elif(hdr['DATA']=='SEG'):
        data = spr_read_segdata(fp,hdr)
    else: #assume LIST data
        data = fp.read()
    return data

def spr_read_segdata(fp,hdr):
    line = fp.readline()

    fshift = 1
    if( hdr['TIMEBASE'] == "CONTINUOUS" ):
        fshift = 0.01
        
    First_time = True
    while line:
        w = line.strip().split()
        if w[0] != "-":
            if First_time:
                First_time = False
                segdata={w[0]:None}
            else:
                segdata[segname] = pd.DataFrame({'t0':t0,'t1':t1,'seg':ww})
            ww = []
            t0 = []
            t1 = []
            cnt = 0
            segname=w[0]

        # process segmentation
        t0.append(round(float(w[2])/fshift))
        t1.append(t0[cnt]+round(float(w[3])/fshift))
        ww.append(w[1])
        cnt+=1
        line = fp.readline()
                                                
    segdata[segname] = pd.DataFrame({'t0':t0,'t1':t1,'seg':ww}) 
    return(segdata)