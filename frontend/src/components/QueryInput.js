// frontend/src/components/QueryInput.js
import React, { useState } from 'react';
import { TextField, Button, Box, CircularProgress, Alert } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

function QueryInput({ onQuerySubmit, isLoading, isDisabled }) {
    const [query, setQuery] = useState('');

    const handleInputChange = (event) => {
        setQuery(event.target.value);
    };

    const handleSubmit = () => {
        if (query.trim()) {
            onQuerySubmit(query);
        } else {
            // In a real app, use a Snackbar or Alert component from MUI
            alert('Please enter a query!');
        }
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !isLoading && query.trim()) {
            handleSubmit();
        }
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
                label="Enter your query"
                variant="outlined"
                fullWidth
                value={query}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                disabled={isLoading || isDisabled}
                placeholder="e.g., What is the main topic of this document?"
                multiline // Allow multiple lines
                rows={3} // Set initial rows
            />
            <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                disabled={isLoading || isDisabled || !query.trim()}
                endIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
                sx={{ alignSelf: 'flex-end' }} // Align button to the right
            >
                Get Answer
            </Button>
            {isDisabled && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                    Please upload a PDF first to enable querying.
                </Alert>
            )}
        </Box>
    );
}

export default QueryInput;