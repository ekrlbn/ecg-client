import numpy as np
from scipy import signal
from tensorflow.keras.models import load_model
import wfdb
import numpy as np
import os
from pathlib import Path
import deepfakeecg

# Constants from the original code
TARGET_SAMPLE_RATE = 500  # Hz
SEGMENT_LENGTH_SECONDS = 10  # seconds
SEGMENT_LENGTH_SAMPLES = SEGMENT_LENGTH_SECONDS * TARGET_SAMPLE_RATE

def preprocess_ecg_signal(ecg_signal, original_sample_rate=None):
    """
    Preprocess a single ECG signal to be compatible with the trained model.

    Parameters:
    -----------
    ecg_signal : numpy.ndarray
        The raw ECG signal
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.

    Returns:
    --------
    processed_signal : numpy.ndarray
        The preprocessed signal ready for model prediction
    """
    # Check for NaN or inf values
    if np.isnan(ecg_signal).any() or np.isinf(ecg_signal).any():
        print("Warning: NaN or Inf values found in signal. Replacing with zeros.")
        ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=1.0, neginf=-1.0)

    # Resample to 500 Hz if needed
    if original_sample_rate is not None and original_sample_rate != TARGET_SAMPLE_RATE:
        # Calculate number of samples after resampling
        num_samples = int(len(ecg_signal) * TARGET_SAMPLE_RATE / original_sample_rate)
        ecg_signal = signal.resample(ecg_signal, num_samples)
        print(f"Resampled signal from {original_sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")

    # Check if signal is long enough
    if len(ecg_signal) < SEGMENT_LENGTH_SAMPLES:
        print(f"Warning: Signal is shorter than required length ({len(ecg_signal)} < {SEGMENT_LENGTH_SAMPLES}).")
        print("Padding with zeros to reach required length.")
        # Pad with zeros
        pad_length = SEGMENT_LENGTH_SAMPLES - len(ecg_signal)
        ecg_signal = np.pad(ecg_signal, (0, pad_length), 'constant')

    # If signal is too long, take the first 10 seconds
    if len(ecg_signal) > SEGMENT_LENGTH_SAMPLES:
        print(f"Signal is longer than required. Taking first {SEGMENT_LENGTH_SECONDS} seconds.")
        ecg_signal = ecg_signal[:SEGMENT_LENGTH_SAMPLES]

    # Z-score normalization
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # Reshape for CNN input [batch, time steps, features]
    processed_signal = ecg_signal.reshape(1, SEGMENT_LENGTH_SAMPLES, 1)

    return processed_signal

def segment_long_ecg(long_ecg, original_sample_rate=None, overlap=0.5):
    """
    Segment a long ECG recording into 10-second segments with overlap.

    Parameters:
    -----------
    long_ecg : numpy.ndarray
        The long ECG recording
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.
    overlap : float
        Overlap between segments (0.5 = 50% overlap)

    Returns:
    --------
    segments : list of numpy.ndarray
        List of preprocessed segments ready for model prediction
    """
    # Resample if needed
    if original_sample_rate is not None and original_sample_rate != TARGET_SAMPLE_RATE:
        # Calculate number of samples after resampling
        num_samples = int(len(long_ecg) * TARGET_SAMPLE_RATE / original_sample_rate)
        resampled_ecg = signal.resample(long_ecg, num_samples)
        print(f"Resampled signal from {original_sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
    else:
        resampled_ecg = long_ecg

    # Clean the signal
    if np.isnan(resampled_ecg).any() or np.isinf(resampled_ecg).any():
        print("Warning: NaN or Inf values found in signal. Replacing with zeros.")
        resampled_ecg = np.nan_to_num(resampled_ecg, nan=0.0, posinf=1.0, neginf=-1.0)

    # Segment the recording into 10-second windows
    window_size = SEGMENT_LENGTH_SAMPLES
    step = int(window_size * (1 - overlap))

    segments = []
    for i in range(0, len(resampled_ecg) - window_size + 1, step):
        segment = resampled_ecg[i:i + window_size]

        # Z-score normalization
        segment = (segment - np.mean(segment)) / np.std(segment)

        # Reshape for CNN input [batch, time steps, features]
        segment = segment.reshape(1, SEGMENT_LENGTH_SAMPLES, 1)
        segments.append(segment)

    return segments

def predict_ecg_authenticity(model, ecg_signal, original_sample_rate=None):
    """
    Predict whether an ECG signal is real or fake.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        The trained classifier model
    ecg_signal : numpy.ndarray
        The raw ECG signal
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.

    Returns:
    --------
    prediction : float
        The model's prediction (0-1, where closer to 1 means more likely fake)
    """
    # Preprocess the signal
    processed_signal = preprocess_ecg_signal(ecg_signal, original_sample_rate)

    # Make prediction
    prediction = model.predict(processed_signal)[0][0]

    return prediction

def classify_ecg_recording(model_path, ecg_signal, original_sample_rate=None, threshold=0.5):
    """
    Complete pipeline to classify an ECG recording as real or fake.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    ecg_signal : numpy.ndarray
        The ECG signal to classify
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.
    threshold : float
        Classification threshold (default 0.5)

    Returns:
    --------
    result : dict
        Dictionary containing classification results
    """
    # Load the model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Determine if it's a long recording or a short segment
    is_long_recording = len(ecg_signal) > (SEGMENT_LENGTH_SAMPLES * 1.5)

    if is_long_recording:
        print(f"Processing long recording ({len(ecg_signal)} samples)")
        # Segment the recording
        segments = segment_long_ecg(ecg_signal, original_sample_rate)

        # Make predictions for each segment
        segment_predictions = []
        for i, segment in enumerate(segments):
            pred = model.predict(segment)[0][0]
            segment_predictions.append(pred)
            print(f"Segment {i+1}/{len(segments)}: {pred:.4f} ({'Fake' if pred > threshold else 'Real'})")

        # Aggregate results
        avg_prediction = np.mean(segment_predictions)
        max_prediction = np.max(segment_predictions)

        result = {
            'classification': 'Fake' if avg_prediction > threshold else 'Real',
            'confidence': abs(avg_prediction - 0.5) * 2,  # Scale to 0-1
            'avg_score': avg_prediction,
            'max_score': max_prediction,
            'segment_predictions': segment_predictions,
            'num_segments': len(segments)
        }
    else:
        # Process as a single segment
        print("Processing single segment")
        processed_signal = preprocess_ecg_signal(ecg_signal, original_sample_rate)
        prediction = model.predict(processed_signal)[0][0]

        result = {
            'classification': 'Fake' if prediction > threshold else 'Real',
            'confidence': abs(prediction - 0.5) * 2,  # Scale to 0-1
            'score': prediction
        }

    return result




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

def read_from_file(file_path: str) -> np.ndarray:
    """
    Read ECG data from .txt file and convert it to a NumPy array.
    
    Parameters:
    -----------
    file_path : str
        Path to the .txt file containing ECG data

    Returns:
    --------
    ecg_signal : numpy.ndarray
        The ECG signal as a NumPy array with shape (n_samples, n_channels)
    
    """
    # Read the .txt file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the data
    data_rows = []
    for line in lines:
        values = [float(val) for val in line.strip().split()]
        if values:  # Skip empty lines
            data_rows.append(values)
    
    # Convert to numpy array
    ecg_signal = np.array(data_rows)
    
    # Verify we have enough data points
    if ecg_signal.shape[0] < 5000:
        print(f"Warning: ECG signal has fewer samples than expected: {ecg_signal.shape[0]}/5000")
    
    print(f"Successfully loaded ECG data from {file_path} with shape {ecg_signal.shape}")
    
    return ecg_signal


def get_prediction_results(data_name, sample_id="100")-> dict:

    model_path = 'ecg_classifier.h5'
    if data_name == "MIT database":
        ecg_signal, fields = read_ecg_signal(sample_id)  # Example record ID
        ecg_signal = ecg_signal[:, 0]  # Use the first channel if multi-channel
        sample_rate = fields['fs']
    else:
        ecg_signal = get_deepfake_ecg()[:,1]  # Example generated signal
        sample_rate = 500

    result = classify_ecg_recording(model_path, ecg_signal, sample_rate)

    # 2. Classify the signal
    # print(f"Classification: {result['classification']}")
    # print(f"Confidence: {result['confidence']:.2f}")
    return result




