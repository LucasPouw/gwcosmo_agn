import os
def get_cachedir(cachedir=None):
    if cachedir is None:
        if 'GWCOSMO_CACHE' in os.environ.keys():
            cachedir = os.environ['GWCOSMO_CACHE']
        else:
            cachedir = os.path.join(os.environ['HOME'], '.cache/gwcosmo')
    os.makedirs(cachedir, exist_ok=True)
    return cachedir
