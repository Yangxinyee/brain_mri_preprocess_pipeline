# Brain MRI Preprocessing Pipeline

A comprehensive automated pipeline for preprocessing stroke MRI data, designed to convert DICOM images to NIfTI format, perform image registration, skull stripping, and organize files for machine learning and analysis workflows.

## Table of Contents

- [Purpose](#purpose)
- [Key Features](#key-features)
- [Environment Setup](#environment-setup)
- [Input Data Format](#input-data-format)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Step-by-Step Execution](#step-by-step-execution)
  - [Command Line Options](#command-line-options)
- [Pipeline Steps](#pipeline-steps)
- [Output Format](#output-format)
- [Error Handling](#error-handling)
- [Troubleshooting](#troubleshooting)

## Purpose

This pipeline is specifically designed for preprocessing stroke MRI datasets, particularly for machine learning applications such as brain lesion segmentation. It addresses common challenges in medical image processing:

- **Multi-modal integration**: Handles DWI, ADC, and FLAIR sequences with intelligent fallback strategies
- **Data standardization**: Converts heterogeneous DICOM data to standardized NIfTI format
- **Spatial alignment**: Registers FLAIR images to DWI space for consistent multi-modal analysis
- **Preprocessing automation**: Performs skull stripping and prepares data for downstream analysis
- **Robust handling**: Manages missing modalities, corrupted files, and edge cases gracefully

## Key Features

### Core Processing Capabilities
- **DICOM Decompression**: Automated decompression of compressed DICOM files
- **Multi-modal Conversion**: Converts DWI, ADC, and FLAIR sequences to NIfTI format
- **Intelligent Modality Handling**: Creates zero-filled NIfTI files for missing modalities
- **Image Registration**: FLAIR-to-DWI registration using ANTs with fallback strategies
- **Skull Stripping**: HD-BET-based brain extraction with GPU acceleration
- **File Organization**: Standardized naming and directory structure

### Advanced Features
- **Edge Case Management**: Handles zero-valued images, missing modalities, and corrupted data
- **Selective Execution**: Run only specific pipeline steps (e.g., conversion only, registration only)
- **Parallel Processing**: Single-worker mode (multi-worker support under development)
- **Comprehensive Logging**: Detailed logging with progress tracking and error reporting
- **Safe File Operations**: Conflict detection and resolution during file operations

### Modality Support Strategies
- **Complete**: All three modalities (DWI, ADC, FLAIR) present
- **DWI Only**: Creates zero-filled ADC and FLAIR
- **FLAIR Only**: Creates zero-filled DWI and ADC
- **Empty DWI**: Uses ADC and FLAIR, creates zero-filled DWI
- **Intelligent Registration**: Adapts registration strategy based on available modalities

## Environment Setup

### System Requirements
- **Operating System**: Linux (recommended), macOS, or Windows
- **Python Version**: 3.10
- **Memory**: 8GB minimum, 16GB recommended for large datasets
- **Storage**: 2-3x input dataset size for intermediate files
- **GPU**: Optional but recommended for HD-BET acceleration

### Software Dependencies

The pipeline requires three main tools. Please refer to their official documentation for detailed installation instructions:

1. **dcm2niix** - DICOM to NIfTI converter
   - Official repository: https://github.com/rordenlab/dcm2niix
   - Installation via conda: `conda install -c conda-forge dcm2niix`

2. **ANTs** - Advanced Normalization Tools for image registration
   - Official repository: https://github.com/ANTsX/ANTs
   - Installation via conda: `conda install -c conda-forge ants`

3. **HD-BET** - Brain extraction tool
   - Official repository: https://github.com/MIC-DKFZ/HD-BET
   - Installation via pip: `pip install hd-bet`

### Environment Setup Steps

1. **Create Virtual Environment**:
   ```bash
   conda create -n brain_mri_env python=3.10
   conda activate brain_mri_env
   ```

2. **Install Dependencies**:
   ```bash
   # Install dcm2niix (refer to official docs for latest version)
   conda install -c conda-forge dcm2niix
   
   # Install ANTs (refer to official docs for latest version)
   conda install -c conda-forge ants
   
   # Install HD-BET (refer to official docs for latest version)
   pip install hd-bet
   
   # Install additional requirements
   pip install nibabel numpy pathlib2 tqdm
   ```

3. **Verify Installation**:
   ```bash
   dcm2niix -h
   antsRegistration --help
   hd-bet -h
   ```

## Input Data Format

### Directory Structure
The pipeline expects input data organized in a hierarchical directory structure:

```
input_directory/
├── Patient_ID_1/
│   ├── Case_ID_1/
│   │   ├── DWI/           # DWI DICOM files
│   │   ├── ADC/           # ADC DICOM files  
│   │   └── FLAIR/         # FLAIR DICOM files
│   └── Case_ID_2/
│       ├── DWI/
│       ├── ADC/
│       └── FLAIR/
└── Patient_ID_2/
    └── Case_ID_3/
        ├── DWI/
        ├── ADC/
        └── FLAIR/
```

### Modality Requirements
- **DWI**: Diffusion-weighted imaging sequences
- **ADC**: Apparent diffusion coefficient maps
- **FLAIR**: Fluid-attenuated inversion recovery sequences

### Supported Scenarios
The pipeline intelligently handles various modality combinations:
- ✅ **Complete cases**: All three modalities present
- ✅ **DWI-only cases**: Only DWI sequences available
- ✅ **FLAIR-only cases**: Only FLAIR sequences available
- ✅ **Missing DWI**: ADC and FLAIR available, DWI missing/corrupted
- ✅ **Zero-valued images**: Handles all-zero images in registration

### File Formats
- **Input**: DICOM files (.dcm, .IMA, or any DICOM format)
- **Intermediate**: NIfTI compressed format (.nii.gz)
- **Output**: Organized NIfTI files with standardized naming

## Usage

### Basic Usage

1. **Activate Environment**:
   ```bash
   conda activate brain_mri_env
   ```

2. **Run Complete Pipeline**:
   ```bash
   python -m brain_mri_preprocess_pipeline.medical_image_pipeline \
     --input /path/to/input/data \
     --output /path/to/output \
     --logs /path/to/logs \
     --workers 1
   ```

   > **⚠️ Important Note**: Currently, only single-worker mode (`--workers 1`) is recommended due to concurrency issues in the skull stripping tool (HD-BET). Multi-worker support is under optimization.

### Step-by-Step Execution

**Run Only Conversion Step**:
```bash
python -m brain_mri_preprocess_pipeline.medical_image_pipeline \
  --input /path/to/input \
  --output /path/to/output \
  --logs /path/to/logs \
  --only-steps conversion
```

**Run Only Registration Step**:
```bash
python -m brain_mri_preprocess_pipeline.medical_image_pipeline \
  --input /path/to/input \
  --output /path/to/output \
  --logs /path/to/logs \
  --only-steps registration
```

**Run Skull Stripping and Subsequent Steps**:
```bash
python -m brain_mri_preprocess_pipeline.medical_image_pipeline \
  --input /path/to/input \
  --output /path/to/output \
  --logs /path/to/logs \
  --only-steps skull_stripping,organization
```

**Run Multiple Specific Steps**:
```bash
python -m brain_mri_preprocess_pipeline.medical_image_pipeline \
  --input /path/to/input \
  --output /path/to/output \
  --logs /path/to/logs \
  --only-steps conversion,registration,skull_stripping
```

### Command Line Options

```
usage: medical_image_pipeline.py [-h] --input INPUT [--output OUTPUT] 
                                [--logs LOGS] [--workers WORKERS] 
                                [--only-steps ONLY_STEPS] [--skip-steps SKIP_STEPS]

Arguments:
  --input INPUT, -i INPUT
                        Input directory containing DICOM files (required)
  --output OUTPUT, -o OUTPUT  
                        Output directory (default: output)
  --logs LOGS, -l LOGS  
                        Log directory (default: logs)  
     --workers WORKERS, -w WORKERS
                         Number of parallel workers (default: 1, recommended: 1)
  --only-steps ONLY_STEPS
                        Run only specified steps (comma-separated)
                        Available: decompression,conversion,registration,skull_stripping,organization
  --skip-steps SKIP_STEPS  
                        Skip specified steps (comma-separated)
```

## Pipeline Steps

The pipeline consists of six main processing steps:

### 1. Decompression
- Decompresses compressed DICOM files
- Preserves directory structure
- Continues processing if some files fail

### 2. Conversion (DICOM → NIfTI)
- Converts DICOM files to NIfTI format using dcm2niix
- Handles modality-specific naming conventions
- Creates zero-filled NIfTI files for missing modalities
- Assigns standardized channel suffixes:
  - `0000`: DWI (Diffusion-weighted imaging)
  - `0001`: ADC (Apparent diffusion coefficient)  
  - `0002`: FLAIR (Fluid-attenuated inversion recovery)

### 3. Registration
- Registers FLAIR images to DWI space using ANTs
- Intelligent strategy selection based on available modalities:
  - **Normal**: DWI → FLAIR registration
  - **Zero DWI**: ADC → FLAIR registration  
  - **Zero FLAIR**: Skip registration (already aligned)
  - **All zero**: Skip registration with warning

### 4. Skull Stripping
- Brain extraction using HD-BET
- GPU acceleration when available
- Processes all modalities consistently

### 5. Organization
- Organizes files into standardized directory structure
- Applies consistent naming convention
- Generates processing statistics

## Output Format

### Directory Structure
```
output_directory/
├── nifti/                           # Converted NIfTI files
│   └── Case_ID/
│       ├── ISLES2022_CaseID_0000.nii.gz    # DWI
│       ├── ISLES2022_CaseID_0001.nii.gz    # ADC  
│       └── ISLES2022_CaseID_0002.nii.gz    # FLAIR
├── skull_stripped/                  # Brain-extracted files
│   └── Case_ID/
│       ├── ISLES2022_CaseID_0000_skull_stripped.nii.gz
│       ├── ISLES2022_CaseID_0001_skull_stripped.nii.gz
│       └── ISLES2022_CaseID_0002_skull_stripped.nii.gz
└── organized/                       # Final organized files
    └── Dataset002_ISLES2022_all/
        └── imagesTs/
            ├── ISLES2022_CaseID_0000.nii.gz
            ├── ISLES2022_CaseID_0001.nii.gz
            └── ISLES2022_CaseID_0002.nii.gz
```

### File Naming Convention
- **Pattern**: `ISLES2022_{CaseID}_{ChannelSuffix}.nii.gz`
- **Channel Suffixes**:
  - `0000`: DWI modality
  - `0001`: ADC modality
  - `0002`: FLAIR modality

## Error Handling

### Robust Processing Features
- **Graceful degradation**: Continues processing when individual cases fail
- **Comprehensive logging**: Detailed error reporting with context
- **Fallback strategies**: Alternative processing methods for edge cases
- **Safe file operations**: Prevents data loss during file conflicts

### Common Error Scenarios
- **Missing modalities**: Creates zero-filled substitutes
- **Corrupted DICOM files**: Skips corrupted files, processes remainder  
- **Registration failures**: Falls back to alternative templates
- **File conflicts**: Safely resolves naming conflicts

## Troubleshooting

### Environment Issues
**Missing Dependencies**:
```bash
# Verify installations
which dcm2niix antsRegistration hd-bet
```

**Python Package Issues**:
```bash
pip install --upgrade nibabel numpy pathlib2 tqdm
```

### Processing Issues
**DICOM Conversion Failures**:
- Check DICOM file integrity
- Verify modality directory structure
- Review conversion logs for specific errors

**Registration Failures**:
- Verify NIfTI files contain valid image data
- Check for all-zero images in logs
- Ensure sufficient disk space for temporary files

**Skull Stripping Issues**:
- Verify HD-BET installation: `hd-bet -h`
- Check GPU availability if using GPU acceleration
- **Always use `--workers 1`** (see parallel processing limitations below)

**Parallel Processing Limitations**:
- **Current Limitation**: Multi-worker processing (`--workers > 1`) is not supported
- **Reason**: The skull stripping tool (HD-BET) has concurrency issues that can cause random failures during parallel execution
- **Recommendation**: Always use `--workers 1` for stable and reliable processing
- **Future**: Multi-worker support is under development and optimization

### Log Analysis
Detailed logs are generated for each step:
- **Main log**: Overall pipeline progress and summary
- **Step-specific logs**: Detailed information for each processing step
- **Error logs**: Specific error messages and stack traces

Example log locations:
```
logs/
├── pipeline_YYYYMMDD_HHMMSS.log      # Main pipeline log
├── conversion_YYYYMMDD_HHMMSS.log    # Conversion step log  
├── registration_YYYYMMDD_HHMMSS.log  # Registration step log
└── skull_stripping_YYYYMMDD_HHMMSS.log # Skull stripping log
```

### Performance Optimization
- **Single worker mode**: **Required** - Currently the only supported mode (`--workers 1`)
- **Parallel processing**: Not supported due to HD-BET concurrency limitations
- **Sufficient disk space**: Ensure 2-3x input size available
- **Memory management**: Monitor RAM usage during processing
- **GPU utilization**: Enable GPU for HD-BET if available for faster skull stripping