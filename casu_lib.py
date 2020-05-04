import psycopg2 as pg2
from psycopg2 import extras
import numpy as np
import astropy.io.fits as pf
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import NoOverlapError
#
def VST_CONNECTION():
    #
    connection=pg2.connect(host="apm53.ast.cam.ac.uk", user="eglez", \
        database="vst", password="casu_db")
    #
    return connection
#
def VISTA_CONNECTION():
    #
    connection=pg2.connect(host="apm53.ast.cam.ac.uk", user="vuser", \
        database="vista", password="dqc_user")
    #
    return connection
#
def query_result_to_dict(cur):
    #
    rel_key = [t[0] for t in cur.description]
    short_result = {}
    results = cur.fetchall()
    for this_key in rel_key:
        temp = []
        for i in range(len(results)):
            temp.append(results[i][this_key])
        temp=np.array(temp)
        short_result[this_key] = temp
    #
    return short_result
#
def find_onchip_vst(ra, dec, stacks = True):
    """
    This function returns all the VST images that   
    contain the given coordinates.

    Parameters
    ----------
    ra : float
        RA to query for, in degrees
    dec : float
        declination to query for, in degrees
    stacks : Bool, optional
        By default, find_onchip queries for stacks, but swithing
        this off queries for tiles. It should be noted that by 
        default, the VST does not produce tiles, so it is likely
        that no images are returned

    Returns
    -------
    dictionary
        A dictionary of arrays, with the filepath, filename, chip and
        filter of all the images containing ra,dec
    """
    #
    conn = VST_CONNECTION()
    #
    # Common bit of the query, precutting by ra,dec
    # because ONCHIP is slow
    #
    query_str = 'WITH temp_table AS ( '
    query_str += 'SELECT * FROM vstqc WHERE '
    query_str += '(ra BETWEEN {0:.3f} AND {1:.3f}) AND '.format(ra-2,ra+2)
    query_str += '(dec BETWEEN {0:.3f} AND {1:.3f}) '.format(dec-2,dec+2)
    query_str += ') '
    #
    if stacks:
        query_str +=' SELECT filename,filepath,chipno,filtname,mjd FROM temp_table WHERE '
        query_str +=' is_stack AND '
        query_str += 'onchip({0:.9f}, {1:.9f}, naxis1, naxis2, crval1, crval2, '.format(ra,dec)
        query_str += 'crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2);'
    else:
        query_str +=' SELECT filename,filepath,chipno,filtname,mjd FROM temp_table WHERE '
        query_str += 'onchip({0:.9f}, {1:.9f}, naxis1, naxis2, crval1, crval2, '.format(ra,dec)
        query_str += 'crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2);'
    #
    cur = conn.cursor(cursor_factory=extras.DictCursor)
    cur.execute(query_str)
    #
    dict_result=query_result_to_dict(cur)
    cur.close()
    conn.close()
    #
    return dict_result
#
def find_onchip_vista(ra, dec, stacks = True):
    """
    This function returns all the VST images that   
    contain the given coordinates.

    Parameters
    ----------
    ra : float
        RA to query for, in degrees
    dec : float
        declination to query for, in degrees
    stacks : Bool, optional
        By default, find_onchip queries for stacks, but swithing
        this off queries for tiles. It should be noted that by 
        default, the VST does not produce tiles, so it is likely
        that no images are returned

    Returns
    -------
    dictionary
        A dictionary of arrays, with the filepath, filename, chip and
        filter of all the images containing ra,dec
    """
    #
    conn = VISTA_CONNECTION()
    #
    # Common bit of the query, precutting by ra,dec
    # because ONCHIP is slow
    #
    query_str = 'WITH temp_table AS ( '
    query_str += 'SELECT * FROM vistaqc WHERE '
    query_str += '(ra BETWEEN {0:.3f} AND {1:.3f}) AND '.format(ra-2,ra+2)
    query_str += '(dec BETWEEN {0:.3f} AND {1:.3f}) '.format(dec-2,dec+2)
    if stacks:
        query_str += 'AND is_stack '
    else:
        query_str += 'AND is_tile '
    query_str += ') '
    #
    if stacks:
        query_str +=' SELECT filename,filepath,chipno,filtname,mjd FROM temp_table WHERE '
        query_str +=' is_stack AND '
        query_str += 'onchip({0:.9f}, {1:.9f}, naxis1, naxis2, crval1, crval2, '.format(ra,dec)
        query_str += 'crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2, pv2_3, pv2_5);'
    else:
        query_str +=' SELECT filename,filepath,chipno,filtname,mjd FROM temp_table WHERE '
        query_str +=' is_tile AND '
        query_str += 'onchip({0:.9f}, {1:.9f}, naxis1, naxis2, crval1, crval2, '.format(ra,dec)
        query_str += 'crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2);'
    #
    cur = conn.cursor(cursor_factory=extras.DictCursor)
    cur.execute(query_str)
    #
    dict_result=query_result_to_dict(cur)
    cur.close()
    conn.close()
    #
    return dict_result
#
def get_cutout(filename,chip,ra,dec,size,return_fits=False):
    """
    This creates a cutout from an input filename. It can be 
    returned either as an array or a fits object that can be 
    saved with return_object.writeto('filename.fits'.

    Parameters
    ----------
    filename : string
        Full path to the image from which the 
        cutout is to be taken.
    chip : int
        Extension in which the object is located
    ra : float
        RA of the cutout centre, in degrees
    dec : float
        declination of the cutout centre, in degrees
    size : float
        size of the side of the cutout, in arcseconds
    return_fits : Boolean, optional
        By default, get_cutout returns the cutout as an
        array, setting this flag to true, it returns an
        HDU fits object with an adequate WCS header.

    Returns
    -------
    array/HDUList
    """
    #
    # Opening image
    #
    hdu=pf.open(filename)
    img_data=hdu[chip].data
    img_wcs=WCS(hdu[chip].header)
    size_in_px=0.5*(size/np.abs(img_wcs.pixel_scale_matrix[0,0]*3600.)+\
                    size/np.abs(img_wcs.pixel_scale_matrix[1,1]*3600.))
    size_in_px=int(round(size_in_px))
    hdu.close()
    # central coordinates to skycoord object
    cc=SkyCoord(ra,dec,unit='deg')
    try:
        cutout=Cutout2D(img_data,cc,size=size_in_px,wcs=img_wcs,mode='trim')
    except NoOverlapError:
        print('Coordinates outside image')
        return []
    #
    if return_fits:
        img_object=pf.PrimaryHDU(cutout.data)
        img_object.header.update(cutout.wcs.to_header())
        img_object.header['PROV1']=(filename.split('/')[-1],'Cutout taken from')
        hdu_object=pf.HDUList([img_object])
        #
        return hdu_object
    else:
        return cutout.data
#
def find_path(filename):
    """
    Returns the filepath to an image file.
    This should be used with care, the database is not 
    indexed by filename, so it will be relatively slow.
    All the files from the same date will be stored in 
    the same location on disc.

    Parameters
    ----------
    filename : string
        image to be located.

    Returns
    -------
    str
    """
    #
    if filename[0]=='v':
        conn = VISTA_CONNECTION()
        db_name='vistaqc'
    if filename[0]=='o':
        conn = VST_CONNECTION()
        db_name='vstqc'
    #
    query_str =' SELECT filepath FROM '+db_name+' WHERE '
    query_str += "filename =\'"+filename+"\' AND chipno=1;"
    cur = conn.cursor(cursor_factory=extras.DictCursor)
    cur.execute(query_str)
    #
    dict_result=query_result_to_dict(cur)
    #
    return dict_result['filepath'][0]
#

