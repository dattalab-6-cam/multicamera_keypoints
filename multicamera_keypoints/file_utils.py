import os
from glob import glob
import warnings
import pdb


class GreaterThanExpectedMatchingFileError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class NoMatchingFilesError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

def filename_from_path(path):
    """Returns just the file name (no extension) from a full path (eg /my/path/to/file.txt returns 'file')

    Arguments:
        path {Path} -- full path

    Returns:
        str -- the file name without extension
    """
    return os.path.splitext(os.path.basename(path))[0]

def to_snake_case(string):
    string = string.strip('\n\r\t ')
    string = string.replace('-', '_')
    string = string.replace(' ', '_')
    string = string.lower()
    return string

def find_files_from_pattern(path, pattern, exclude_patterns=None, n_expected=1, error_behav='raise'):
    """Find a given number of files on a given path, matching a particular glob pattern. 
    Raises an error if wrong number of files match, unless error_behav='pass'.

    Arguments:
        path {str} -- path to check
        pattern {str} -- passed to glob.glob
    Keyword Arguments:
        exclude_patterns {list(str)} -- if any of these patterns is in found file names, exclude them
        n_expected {int} -- number of files expected to match pattern (default: {1})
        error_behav {str} -- if 'raise', raises errors; if 'pass', return None for no files, or the whole list for multiple.
    Raises:
        DataProcessingError -- if more than one file matching the pattern exists

    Returns:
        str -- path to matching file (or None if no file)
    """
    if exclude_patterns is None:
        exclude_patterns = []
    elif type(exclude_patterns) is str:
        exclude_patterns = [exclude_patterns]

    if "*" not in pattern:
        warnings.warn(
            "It appears your glob pattern has no wildcard, are you sure that's right?"
        )

    potential_file_list = glob(f"{path}/{pattern}")

    # Raise error if no files found
    if len(potential_file_list) == 0:
        if error_behav=='raise':
            raise NoMatchingFilesError(f"Found zero files matching {path}/{pattern}!")
        else:
            return None

    # Exclude requested patterns
    for exclude_pattern in exclude_patterns:
        potential_file_list = [
            pattern for pattern in potential_file_list if exclude_pattern not in os.path.basename(pattern)
        ]

    # If now no files, they were removed via pattern, so stop
    if len(potential_file_list) == 0:
        warnings.warn(f"No files remaining in {path} after exclusion.")
        if error_behav=='raise':
            raise NoMatchingFilesError(f"Found zero files matching {path}/{pattern}!")
        else:
            return None

    # Raise error if still more than one matching file
    if len(potential_file_list) > n_expected:
        if error_behav=='raise':
            raise GreaterThanExpectedMatchingFileError(
                f"Found {len(potential_file_list)} files matching {path}/{pattern} but expected {n_expected}!"
            )
        else:
            return potential_file_list

    if n_expected == 1:
        return potential_file_list[0]
    else:
        return potential_file_list
