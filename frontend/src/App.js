// frontend/src/App.js
import React, { useState } from 'react';
import PdfUploader from './components/PdfUploader';
import QueryInput from './components/QueryInput';
import ResultsDisplay from './components/ResultsDisplay';
import { Container, Typography, Box } from '@mui/material'; // Import MUI components
import './App.css'; // Keep custom CSS for overall layout

function App() {
  const [processingPdf, setProcessingPdf] = useState(false);
  const [pdfUploadStatus, setPdfUploadStatus] = useState({ message: '', type: '' });
  const [querying, setQuerying] = useState(false);
  const [queryStatus, setQueryStatus] = useState({ message: '', type: '' });
  const [answer, setAnswer] = useState('');
  const [chunks, setChunks] = useState([]);
  const [pdfProcessed, setPdfProcessed] = useState(false);

  const handlePdfUpload = async (file) => {
    setProcessingPdf(true);
    setPdfUploadStatus({ message: 'Uploading and processing PDF... This may take a moment.', type: 'loading' });
    setPdfProcessed(false);
    setAnswer('');
    setChunks([]);

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
    setAnswer('');
    setChunks([]);

    try {
      const response = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userQuery }),
      });
      const data = await response.json();

      if (response.ok) {
        setQueryStatus({ message: '', type: '' });
        setAnswer(data.answer);
        setChunks(data.chunks || []);
      } else {
        setQueryStatus({ message: `Error: ${data.error || 'Unknown error during query.'}`, type: 'error' });
        setAnswer('No answer could be retrieved.');
      }
    } catch (error) {
      setQueryStatus({ message: `Network error: ${error.message}`, type: 'error' });
      setAnswer('No answer could be retrieved due to a network error.');
    } finally {
      setQuerying(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}> {/* Use MUI Container for centering and max-width */}
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
        <ResultsDisplay answer={answer} chunks={chunks} />
      </Box>
    </Container>
  );
}

export default App;