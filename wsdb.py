#!/usr/bin/env python

import numpy as np
import sqlutil as sqlutilpy
from astropy.io import fits
import getpass


class connCache:
    # connection cache
    conn = None

    
def connect(username=''):
    # connect to wsdb using the supplied username
    # you will be prompted for the password but you 
    # can hardcode below if you like (and if you 
    # set appropriate permissions to this file)
    if username == 'colin_mao':
        with open('wsdb.txt') as f:
            password = f.read()
    else:
        password = getpass.getpass()
    
    connCache.conn = sqlutilpy.getConnection(
        db='wsdb',
        driver="psycopg2",
        host='cappc127.ast.cam.ac.uk',
        user=username,
        password=password)
    return


def getsql(sql, conn=None):
    # run a query, return the result as an ordered dict
    if conn is None:
        if connCache.conn is None:
            conn = connect()
            connCache.conn = conn
        else:
            conn = connCache.conn
    return sqlutilpy.get(sql, conn=conn, asDict=True)


def execsql(sql, conn=None):
    # run a query with no result
    # useful for performing server operations that
    # don't return a table as a response, e.g. 
    # altering table permissions
    if conn is None:
        if connCache.conn is None:
            conn = connect()
            connCache.conn = conn
        else:
            conn = connCache.conn
    return sqlutilpy.execute(sql, conn=conn)


def sql2npy(sql, conn=None):
    # run a query and return the table as a numpy
    # recarray with correctly named fields
    data = getsql(sql, conn)
    
    if len(data) == 0:
        return np.array([])

    names = []
    dtypes = []
    for key, value in data.items():
        names += [key]
        dtypes += [value.dtype]

    op = np.empty(data[names[0]].size, dtype={'names':names,'formats':dtypes})

    for key, value in data.items():
        op[key] = value

    return op


def sql2fits(sql, filename, conn=None):
    # run a query and write the result to disk as 
    # a FITS format binary table file
    op = sql2npy(sql, conn)
    fits.writeto(filename, op)
