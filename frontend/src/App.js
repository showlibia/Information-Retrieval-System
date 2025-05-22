// frontend/src/App.js
import React, { useState } from 'react'; // Removed useEffect import
import PdfUploader from './components/PdfUploader';
import QueryInput from './components/QueryInput';
import ResultsDisplay from './components/ResultsDisplay';
import { Container, Typography, Box, FormControlLabel, Switch } from '@mui/material';
import './App.css';

function App() {
  const [processingPdf, setProcessingPdf] = useState(false);
  const [pdfUploadStatus, setPdfUploadStatus] = useState({ message: '', type: '' });
  const [querying, setQuerying] = useState(false);
  const [queryStatus, setQueryStatus] = useState({ message: '', type: '' });
  const [answer, setAnswer] = useState(''); // This will only hold LLM answer
  const [chunks, setChunks] = useState([]); // This will hold the top 5 ranked chunks
  const [pdfProcessed, setPdfProcessed] = useState(false);
  const [useLLM, setUseLLM] = useState(false);

  // Removed the useEffect that constructed 'answer' from chunks.
  // The 'answer' area will be empty or show LLM results.
  // The ranked chunks will be displayed in ResultsDisplay's own section.


  const handlePdfUpload = async (file) => {
    setProcessingPdf(true);
    setPdfUploadStatus({ message: 'Uploading and processing PDF... This may take a moment.', type: 'loading' });
    setPdfProcessed(false);
    setAnswer(''); // Clear answer on new PDF upload
    setChunks([]); // Clear chunks on new PDF upload

    const formData = new FormData();
    formData.append('pdf_file', file);

    try {
      const response = await fetch('http://localhost:5000/upload_pdf', {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });
      const data = await response.json();

      if (response.ok && data.success) {
        setPdfUploadStatus({ message: data.message, type: 'success' });
        setPdfProcessed(true);
      } else {
        setPdfUploadStatus({ message: `Error: ${data.error || 'Unknown error during upload.'}`, type: 'error' });
      }
    } catch (error) {
      setPdfUploadStatus({ message: `Network error: ${error.message}`, type: 'error' });
    } finally {
      setProcessingPdf(false);
    }
  };

  const handleQuery = async (userQuery) => {
    setQuerying(true);
    setQueryStatus({ message: 'Getting answer from document...', type: 'loading' });
    setAnswer(''); // Clear previous answer
    setChunks([]); // Clear previous chunks

    try {
      const response = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userQuery, use_llm: useLLM }),
        credentials: 'include'
      });
      const data = await response.json();

      if (response.ok) {
        setQueryStatus({ message: '', type: '' });
        setChunks(data.chunks || []); // Always set top N chunks from backend response

        if (useLLM) {
          setAnswer(data.answer); // LLM-generated answer
        } else {
          setAnswer(''); // When LLM is off, the main answer area remains blank for the screenshot's effect
        }

      } else {
        setQueryStatus({ message: `Error: ${data.error || 'Unknown error during query.'}`, type: 'error' });
        setAnswer('No answer could be retrieved.');
        setChunks([]);
      }
    } catch (error) {
      setQueryStatus({ message: `Network error: ${error.message}`, type: 'error' });
      setAnswer('No answer could be retrieved due to a network error.');
      setChunks([]);
    } finally {
      setQuerying(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        PDF Information Retrieval System
      </Typography>

      <Box sx={{ bgcolor: '#fff', p: 3, borderRadius: 2, boxShadow: 3, mb: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          1. Upload PDF
        </Typography>
        <PdfUploader onFileUpload={handlePdfUpload} isLoading={processingPdf} />
        {pdfUploadStatus.message && (
          <Typography
            variant="body2"
            sx={{ mt: 2, p: 1, borderRadius: 1 }}
            className={`status-message ${pdfUploadStatus.type}`}
          >
            {pdfUploadStatus.message}
          </Typography>
        )}
      </Box>

      <Box sx={{ bgcolor: '#fff', p: 3, borderRadius: 2, boxShadow: 3, mb: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          2. Ask a Question
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={useLLM}
                onChange={(event) => setUseLLM(event.target.checked)}
                name="useLLMToggle"
                color="primary"
                disabled={querying}
              />
            }
            label="Use LLM for Answer Generation"
          />
        </Box>
        <QueryInput onQuerySubmit={handleQuery} isLoading={querying} isDisabled={!pdfProcessed} />
        {queryStatus.message && (
          <Typography
            variant="body2"
            sx={{ mt: 2, p: 1, borderRadius: 1 }}
            className={`status-message ${queryStatus.type}`}
          >
            {queryStatus.message}
          </Typography>
        )}
      </Box>

      <Box sx={{ bgcolor: '#fff', p: 3, borderRadius: 2, boxShadow: 3, mb: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Response
        </Typography>
        {/* Pass answer, chunks, and useLLM to ResultsDisplay */}
        <ResultsDisplay answer={answer} chunks={chunks} useLLM={useLLM} />
      </Box>
    </Container>
  );
}

export default App;