import numpy as np
import wfdb
import numpy as np
import os
from pathlib import Path
import deepfakeecg

# Constants from the original code
TARGET_SAMPLE_RATE = 500  # Hz
SEGMENT_LENGTH_SECONDS = 10  # seconds
SEGMENT_LENGTH_SAMPLES = SEGMENT_LENGTH_SECONDS * TARGET_SAMPLE_RATE


def read_ecg_signal(record_id, data_dir='mit-bih-arrhythmia-database-1.0.0'):
    """
    Read an ECG signal from the MIT-BIH Arrhythmia Database and convert it to a NumPy array.
    
    Parameters:
    -----------
    record_id : str or int
        The record ID of the ECG signal (e.g., '100', '101', etc.)
    data_dir : str, optional
        The directory containing the MIT-BIH database files
        
    Returns:
    --------
    signals : numpy.ndarray
        The ECG signals as a NumPy array with shape (num_samples, num_channels)
    fields : dict
        Metadata about the signal (sampling frequency, units, etc.)
    
    Example:
    --------
    >>> signals, fields = read_ecg_signal('100')
    >>> print(f"Signal shape: {signals.shape}")
    >>> print(f"Sampling frequency: {fields['fs']} Hz")
    >>> print(f"Signal names: {fields['sig_name']}")
    """
    # Convert record_id to string if it's not already
    record_id = str(record_id)
    
    # Create full path to the record
    record_path = os.path.join(data_dir, record_id)
    
    try:
        # Read the record using wfdb
        record = wfdb.rdrecord(record_path)
        
        # Extract signals as numpy array
        signals = np.array(record.p_signal)
        
        # Create a dictionary of metadata fields
        fields = {
            'fs': record.fs,  # Sampling frequency
            'n_sig': record.n_sig,  # Number of signals
            'sig_name': record.sig_name,  # Signal names
            'units': record.units,  # Signal units
            'adc_gain': record.adc_gain,  # ADC gain values
            'comments': record.comments if hasattr(record, 'comments') else [],  # Comments
            'n_samples': record.sig_len  # Number of samples
        }
        
        return signals, fields
    
    except Exception as e:
        print(f"Error reading record {record_id}: {e}")
        return None, None


def list_available_records(data_dir='mit-bih-arrhythmia-database-1.0.0'):
    """
    List all available record IDs in the MIT-BIH Arrhythmia Database.
    
    Parameters:
    -----------
    data_dir : str, optional
        The directory containing the MIT-BIH database files
        
    Returns:
    --------
    list
        A list of available record IDs
    """
    try:
        # Look for .hea files (header files) which indicate available records
        data_path = Path(data_dir)
        record_files = list(data_path.glob('*.hea'))
        record_ids = [file.stem for file in record_files]
        return sorted(record_ids)
    except Exception as e:
        print(f"Error listing records: {e}")
        return []

def get_deepfake_ecg() -> np.ndarray:
    """
    Generate an ECG signal using deepfakeecg library and read it from .asc file.
    Returns:
    --------
    ecg_signal : numpy.ndarray
        The generated ECG signal as a NumPy array with shape (n_samples, 8)
        representing 8 leads [I, II, V1, V2, V3, V4, V5, V6]
    """
    deepfakeecg.generate(1, "./deepfake_ecg/", start_id=0)
    
    # Load the generated ECG file (.asc format)
    ecg_file_path = "./deepfake_ecg/0.asc"
    if not os.path.exists(ecg_file_path):
        raise FileNotFoundError(f"Generated ECG file not found at {ecg_file_path}")
    
    # Read the .asc file
    # ASC files typically have a header followed by data
    # We'll implement a flexible parser to handle different formats
    with open(ecg_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find where the data starts (after header)
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and all(c.isdigit() or c in '.-+eE,' for c in line.strip()):
            data_start_idx = i
            break
    
    # Parse the data
    data_lines = lines[data_start_idx:]
    data_rows = []
    
    # Determine if data is space or comma separated
    delimiter = ',' if ',' in data_lines[0] else None  # None means whitespace
    
    for line in data_lines:
        values = [float(val) for val in line.strip().split(delimiter)]
        if values:  # Skip empty lines
            data_rows.append(values)
    
    # Convert to numpy array
    raw_data = np.array(data_rows)
    
    # Handle different possible data layouts
    if raw_data.shape[1] == 8:
        # Data already in 8-lead format
        ecg_signal = raw_data
    elif raw_data.shape[1] > 8:
        # More columns than needed, take first 8 (assuming leads are in order)
        ecg_signal = raw_data[:, :8]
        print(f"Warning: .asc file has {raw_data.shape[1]} columns, using first 8 for leads")
    else:
        raise ValueError(f"Not enough columns in .asc file. Expected at least 8, got {raw_data.shape[1]}")
        
    # Verify we have enough data points
    if ecg_signal.shape[0] < 5000:
        print(f"Warning: ECG signal has fewer samples than expected: {ecg_signal.shape[0]}/5000")
    
    print(f"Successfully loaded 8-lead ECG data with shape {ecg_signal.shape}")
    
    return ecg_signal
