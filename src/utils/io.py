import matplotlib.pyplot as plt
import os
from IPython.display import display, FileLink
from datetime import datetime
import joblib

from src.config.settings import PATH_OUT_VISUALS
from src.utils.logging import get_logger

logger = get_logger()

def save_and_show_link(fig_to_save, filename, base_dir=PATH_OUT_VISUALS, dpi=100, show_link: bool = True):
    """
    Save a matplotlib figure to disk and optionally show a clickable link.

    Parameters
    ----------
    fig_to_save : matplotlib.figure.Figure
    filename : str
        Output filename including extension.
    base_dir : str or Path
        Destination directory. Defaults to PATH_OUT_VISUALS.
    dpi : int
        Resolution. Defaults to 100.
    show_link : bool
        If True, renders a FileLink in the notebook output.
    """
    logger.debug("START ...")
    os.makedirs(base_dir, exist_ok=True)

    # Get absolute paths
    notebook_dir = os.getcwd()
    full_base_dir = os.path.abspath(os.path.join(notebook_dir, base_dir))
    full_filepath = os.path.join(full_base_dir, filename)

    print(f"Saving figure to {full_filepath}")

    # Save the figure with high DPI
    fig_to_save.savefig(full_filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig_to_save)  # Close the figure to free memory

    if show_link:
        display(FileLink(full_filepath))        # Display a link to the saved figure

    logger.debug("... FINISH")

def get_current_timestamp():
    """
    Return the current datetime as a compact string (YYYYmmdd_HHMMSS_mmm).

    Returns
    -------
    str
    """
    logger.debug("START ...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    logger.debug("... FINISH")
    return timestamp

def save_file(file_type_to_save, filename, base_dir_path, data):
    """
    Save data to disk in the format matching the given file type.

    Supported types are 'feature', 'model', 'hyperparams', and 'metrics'.
    For 'metrics', content is appended if the file already exists.

    Parameters
    ----------
    file_type_to_save : str
        One of 'feature', 'model', 'hyperparams', 'metrics'.
    filename : str
        Output filename.
    base_dir_path : str or Path
        Destination directory.
    data : object
        Content to save; type depends on file_type_to_save.
    """
    logger.debug("START ...")
    os.makedirs(base_dir_path, exist_ok=True)

    # Get absolute paths
    notebook_dir = os.getcwd()
    full_base_dir = os.path.abspath(os.path.join(notebook_dir, base_dir_path))
    full_filepath = os.path.join(full_base_dir, filename)

    if file_type_to_save == 'feature':
        logger.info(f'Saving {file_type_to_save} file with filename as {filename} to path {full_filepath}')
        data.to_csv(full_filepath, index=False, header=False)
    elif file_type_to_save == 'model':
        logger.info(f'Saving {file_type_to_save} file with filename as {filename} to path {full_filepath}')
        joblib.dump(data, full_filepath)
    elif file_type_to_save == 'hyperparams':
        logger.info(f'Saving {file_type_to_save} file with filename as {filename} to path {full_filepath}')
        data.to_csv(full_filepath, index=False)
    elif file_type_to_save == 'metrics':
        logger.info(f'Saving metrics into file {filename} at path {full_filepath}')
        # Check if a file exists
        if os.path.exists(full_filepath):
            # File exists, append content
            with open(full_filepath, 'a') as file:
                file.write('\n' + data)
            print(f"Content appended to existing file: {full_filepath}")
        else:
            # File doesn't exist, create and write content
            with open(full_filepath, 'w') as file:
                file.write(data)
            print(f"New file created with content: {full_filepath}")
    else:
        logger.debug(f'No matching FILE TYPE found for: {file_type_to_save}')

    logger.debug("... FINISH")