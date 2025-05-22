// frontend/src/components/PdfUploader.js
import React, { useState } from 'react';
import { Button, TextField, Box, CircularProgress, InputAdornment } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';

function PdfUploader({ onFileUpload, isLoading }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleSubmit = () => {
        if (selectedFile) {
            onFileUpload(selectedFile);
        } else {
            // In a real app, use a Snackbar or Alert component from MUI
            alert('Please select a PDF file first!');
        }
    };

    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            {/* Hidden input for file selection */}
            <input
                accept=".pdf"
                style={{ display: 'none' }}
                id="raised-button-file"
                multiple
                type="file"
                onChange={handleFileChange}
                disabled={isLoading}
            />
            <label htmlFor="raised-button-file">
                <Button
                    variant="contained"
                    component="span" // important for the button to act as a label for the input
                    startIcon={<UploadFileIcon />}
                    disabled={isLoading}
                >
                    Choose File
                </Button>
            </label>
            <TextField
                variant="outlined"
                size="small"
                value={selectedFile ? selectedFile.name : 'No file chosen'}
                fullWidth
                readOnly
                sx={{ flexGrow: 1 }}
                InputProps={{
                    readOnly: true,
                    startAdornment: selectedFile && (
                        <InputAdornment position="start">
                            <UploadFileIcon />
                        </InputAdornment>
                    ),
                }}
            />
            <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                disabled={isLoading || !selectedFile}
                endIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : null}
            >
                Upload and Process PDF
            </Button>
        </Box>
    );
}

export default PdfUploader;