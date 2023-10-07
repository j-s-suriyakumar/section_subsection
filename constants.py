CREDS = [' MD ', ' DO ', ' DPM ', ' PA ', ' PA-C ', ' CRNA ', ' RN ',
         ' D.O. ', ' PHD ', ' MBBS ', ' NP ', ' APRN ', ' DNP-APRN ']
CREDS.sort(key=len, reverse=True)

SECTION_END_PATTERNS = ['electronically witnessed', 'electronically signed', 'addendum by', 
                        'addendum', 'signed by', 'added by', 'edited by','authenticated by',
                        'consults by', 'reviewed by', 'transcribv by', 'transcribe by', 'review by', 
                        'sign by', 'verify by', 'perform', 'signed on']
SECTION_END_PATTERNS.sort(key=len, reverse=True)

REPLACE_STRINGS = '(continued)'

