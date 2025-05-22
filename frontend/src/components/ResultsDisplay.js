// frontend/src/components/ResultsDisplay.js
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
    Typography, Paper, Box, List, ListItem, ListItemText,
    Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'; // Import icon for accordion

function ResultsDisplay({ answer, chunks, useLLM, otherMethodChunks }) { // Added otherMethodChunks prop
    const [expanded, setExpanded] = useState(false); // State for accordion expansion

    const handleChange = (panel) => (event, isExpanded) => {
        setExpanded(isExpanded ? panel : false);
    };

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
                        {useLLM ? "Generating answer with LLM..." : "Upload a PDF and ask a question to get started, or toggle LLM on for a summarized response."}
                    </Typography>
                )}
            </Paper>

            {/* Top Ranked Chunks - Primary display */}
            {chunks.length > 0 && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="h6" component="h3" gutterBottom>
                        Top Ranked Chunks:
                    </Typography>
                    {chunks.map((chunk, index) => (
                        <Paper
                            key={chunk.doc_id + '-' + index} // Unique key including doc_id and index
                            elevation={1}
                            sx={{
                                p: 2,
                                mb: 1, // Reduced margin for closer boxes
                                bgcolor: '#e9f7ef',
                                border: '1px solid #d4edda',
                                display: 'flex',
                                flexDirection: 'column',
                                '& .MuiTypography-root': {
                                    wordBreak: 'break-word',
                                },
                            }}
                        >
                            <Typography variant="subtitle2" component="div" sx={{ fontWeight: 'bold' }}>
                                {/* Display all query methods that found this chunk */}
                                {`[${chunk.query_methods.map(method => method.replace('_', ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')).join(', ')}] `}
                                <span style={{ color: '#555' }}>{`(Score: ${chunk.score}, Doc: ${chunk.doc}, Page: ${chunk.page})`}</span>
                            </Typography>
                            <Typography variant="body2" sx={{ mt: 0.5 }}>
                                {chunk.text}
                            </Typography>
                        </Paper>
                    ))}
                </Box>
            )}

            {/* Other Retrieval Methods - Accordion Section */}
            {otherMethodChunks && otherMethodChunks.length > 0 && ( // Ensure otherMethodChunks exists and has content
                <Accordion expanded={expanded === 'panel1'} onChange={handleChange('panel1')} sx={{ mt: 2 }}>
                    <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls="panel1bh-content"
                        id="panel1bh-header"
                    >
                        <Typography variant="h6" component="h3" sx={{ width: '33%', flexShrink: 0 }}>
                            Other Retrieval Methods (Additional Chunks)
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <List dense>
                            {otherMethodChunks.map((chunk, index) => (
                                <ListItem key={chunk.doc_id + '-' + index + '-other'} divider={index < otherMethodChunks.length - 1}>
                                    <ListItemText
                                        primary={chunk.text}
                                        secondary={`Doc: ${chunk.doc}, Page: ${chunk.page} | Methods: ${chunk.query_methods.join(', ')} | Score: ${chunk.score}`}
                                        primaryTypographyProps={{ variant: 'body2' }}
                                        secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                                    />
                                </ListItem>
                            ))}
                        </List>
                    </AccordionDetails>
                </Accordion>
            )}
            {/* Display message if no other chunks */}
            {(!otherMethodChunks || otherMethodChunks.length === 0) && (
                <Accordion expanded={expanded === 'panel1'} onChange={handleChange('panel1')} sx={{ mt: 2 }}>
                    <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls="panel1bh-content"
                        id="panel1bh-header"
                    >
                        <Typography variant="h6" component="h3" sx={{ width: '33%', flexShrink: 0 }}>
                            Other Retrieval Methods (Additional Chunks)
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Typography variant="body2" color="text.secondary">
                            No additional chunks from other methods to display (all relevant chunks might be in Top Ranked Chunks).
                        </Typography>
                    </AccordionDetails>
                </Accordion>
            )}
        </Box>
    );
}

export default ResultsDisplay;