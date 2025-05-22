// frontend/src/components/ResultsDisplay.js
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
    Typography, Paper, Box, List, ListItem, ListItemText,
    Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'; // Import icon for accordion

function ResultsDisplay({ answer, chunks, useLLM }) {
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

            {/* Top 5 Ranked Chunks - mimicking the green boxes if LLM is OFF */}
            {!useLLM && chunks.length > 0 && (
                <Box sx={{ mb: 3 }}>
                    {chunks.map((chunk, index) => (
                        <Paper
                            key={index}
                            elevation={1}
                            sx={{
                                p: 2,
                                mb: 1, // Reduced margin for closer boxes
                                bgcolor: '#e9f7ef',
                                border: '1px solid #d4edda',
                                display: 'flex',
                                flexDirection: 'column',
                                // Align text content with padding from the example image
                                '& .MuiTypography-root': {
                                    wordBreak: 'break-word',
                                },
                            }}
                        >
                            <Typography variant="subtitle2" component="div" sx={{ fontWeight: 'bold' }}>
                                {/* Example: Top K Search, Boolean Query, Synonym Query, Global Search */}
                                {`[${chunk.query_method.replace('_', ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}] `}
                                <span style={{ color: '#555' }}>{`(Score: ${chunk.score}, Doc: ${chunk.doc}, Page: ${chunk.page})`}</span>
                            </Typography>
                            <Typography variant="body2" sx={{ mt: 0.5 }}>
                                {chunk.text}
                            </Typography>
                        </Paper>
                    ))}
                </Box>
            )}

            {/* Retrieved Chunks - Always visible as a list */}
            <Typography variant="h6" component="h3" gutterBottom>
                Retrieved Chunks:
            </Typography>
            <Paper elevation={1} sx={{ maxHeight: 250, overflowY: 'auto', p: 2, bgcolor: '#fff', mb: 3 }}>
                {chunks.length > 0 ? (
                    <List dense>
                        {chunks.map((chunk, index) => (
                            <ListItem key={index} divider={index < chunks.length - 1}>
                                <ListItemText
                                    primary={chunk.text}
                                    secondary={`Doc: ${chunk.doc}, Page: ${chunk.page}, Sentence: ${chunk.sentence_index} | Method: ${chunk.query_method} | Score: ${chunk.score}`}
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

            {/* Other Retrieval Methods - Accordion Section */}
            <Accordion expanded={expanded === 'panel1'} onChange={handleChange('panel1')} sx={{ mt: 2 }}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="panel1bh-content"
                    id="panel1bh-header"
                >
                    <Typography variant="h6" component="h3" sx={{ width: '33%', flexShrink: 0 }}>
                        Other Retrieval Methods (Top 3 Chunks Each)
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    {/*
                      Since backend only returns top 5 overall, we'll just re-display the top 3
                      of the existing chunks here to simulate the accordion's content.
                      In a real scenario, the backend would provide a breakdown by method.
                    */}
                    {chunks.slice(0, 3).length > 0 ? (
                        <List dense>
                            {chunks.slice(0, 3).map((chunk, index) => (
                                <ListItem key={index} divider={index < chunks.slice(0, 3).length - 1}>
                                    <ListItemText
                                        primary={chunk.text}
                                        secondary={`Doc: ${chunk.doc}, Page: ${chunk.page}, Sentence: ${chunk.sentence_index} | Method: ${chunk.query_method} | Score: ${chunk.score}`}
                                        primaryTypographyProps={{ variant: 'body2' }}
                                        secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                                    />
                                </ListItem>
                            ))}
                        </List>
                    ) : (
                        <Typography variant="body2" color="text.secondary">
                            No additional chunks from other methods to display.
                        </Typography>
                    )}
                </AccordionDetails>
            </Accordion>
        </Box>
    );
}

export default ResultsDisplay;