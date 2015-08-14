# coding: utf8

DOIT_CONFIG = {'default_tasks': []}

CITEULIKE_GROUP = 19226
BIBFILE = 'docs/pypercolate.bib'


def task_download_bib():
    """Download bibliography from CiteULike group"""

    return {
        'actions': [' '.join([
            'wget', '-O', BIBFILE,
            '"http://www.citeulike.org/bibtex/group/{}?incl_amazon=0&key_type=4"'.format(CITEULIKE_GROUP),
            ])],
        # 'file_dep': [CITEULIKE_COOKIES],
        'targets': [BIBFILE],
    }
