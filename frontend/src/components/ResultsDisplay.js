// frontend/src/components/ResultsDisplay.js
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Typography, Paper, Box, List, ListItem, ListItemText } from '@mui/material';

function ResultsDisplay({ answer, chunks }) {
    return (
        <Box>
            <Typography variant="h6" component="h3" gutterBottom>
                Answer:
            </Typography>
            <Paper elevation={1} sx={{ p: 2, mb: 3, bgcolor: '#e9f7ef', border: '1px solid #d4edda' }}>
                {answer ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {answer}
                    </ReactMarkdown>
                ) : (
                    <Typography variant="body1" color="text.secondary">
                        Upload a PDF and ask a question to get started.
                    </Typography>
                )}
            </Paper>

            <Typography variant="h6" component="h3" gutterBottom>
                Retrieved Chunks:
            </Typography>
            <Paper elevation={1} sx={{ maxHeight: 250, overflowY: 'auto', p: 2, bgcolor: '#fff' }}>
                {chunks.length > 0 ? (
                    <List dense> {/* dense makes list items smaller */}
                        {chunks.map((chunk, index) => (
                            <ListItem key={index} divider={index < chunks.length - 1}>
                                <ListItemText
                                    primary={chunk.text}
                                    secondary={`Doc: ${chunk.doc}, Page: ${chunk.page}`}
                                    primaryTypographyProps={{ variant: 'body2' }}
                                    secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                                />
                            </ListItem>
                        ))}
                    </List>
                ) : (
                    <Typography variant="body2" color="text.secondary">
                        Relevant text chunks will appear here.
                    </Typography>
                )}
            </Paper>
        </Box>
    );
}

export default ResultsDisplay;