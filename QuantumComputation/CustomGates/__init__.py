
def _import_all_modules():
    """ Dynamically imports all modules in this package. """
    import traceback
    import os
    global __all__
    __all__ = []
    globals_, locals_ = globals(), locals()

    # Dynamically import all the package modules in this file's directory.
    for filename in os.listdir(__name__):
        # Process all python files in directory that don't start
        # with underscore (which also prevents this module from
        # importing itself).
        if filename[0] != '_' and filename.split('.')[-1] in ('py', 'pyw'):
            modulename = filename.split('.')[0]  # Filename sans extension.
            package_module = '.'.join([__name__, modulename])
            
            globals_[modulename] = modulename
            __all__.append(modulename)
            
            try:
                module = __import__(package_module, globals_, locals_, [modulename])
            except:
                traceback.print_exc()
                raise
            
            

_import_all_modules()
