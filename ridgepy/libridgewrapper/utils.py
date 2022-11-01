def wrap_function(library, function_name, result_type, argument_types):
    function          = library.__getattr__(function_name)
    function.restype  = result_type
    function.argtypes = argument_types
    return function
