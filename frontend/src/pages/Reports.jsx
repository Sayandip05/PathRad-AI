import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Checkbox,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  PictureAsPdf as PdfIcon,
  TextSnippet as TextIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { getDiagnoses, generateReport, downloadReport } from '../services/api';

function Reports() {
  const [diagnoses, setDiagnoses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDiagnosis, setSelectedDiagnosis] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [includeImages, setIncludeImages] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [reportFormat, setReportFormat] = useState('pdf');

  useEffect(() => {
    fetchCompletedDiagnoses();
  }, []);

  const fetchCompletedDiagnoses = async () => {
    try {
      setLoading(true);
      const response = await getDiagnoses();
      const completed = response.data.filter(d => d.status === 'completed');
      setDiagnoses(completed);
    } catch (err) {
      setError('Failed to load diagnoses');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!selectedDiagnosis) return;

    try {
      setGenerating(true);
      const response = await generateReport(selectedDiagnosis.id, includeImages, reportFormat);

      const downloadResponse = await downloadReport(
        response.data.report_path.split('/').pop()
      );

      const url = window.URL.createObjectURL(new Blob([downloadResponse.data]));
      const link = document.createElement('a');
      link.href = url;

      const extension = reportFormat === 'txt' ? 'txt' : 'pdf';
      link.setAttribute('download', `PathRad_Report_${selectedDiagnosis.patient?.case_id}.${extension}`);

      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      setDialogOpen(false);
    } catch (err) {
      setError('Failed to generate report');
      console.error(err);
    } finally {
      setGenerating(false);
    }
  };

  const openGenerateDialog = (diagnosis) => {
    setSelectedDiagnosis(diagnosis);
    setDialogOpen(true);
  };

  const handleFormatChange = (event, newFormat) => {
    if (newFormat !== null) {
      setReportFormat(newFormat);
    }
  };

  if (loading) {
    return (
      <Box sx={{ width: '100%', mt: 4 }}>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Reports
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Typography variant="body1" paragraph>
        Generate and download diagnostic reports for completed cases. Reports available in PDF and TXT formats.
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Case ID</TableCell>
              <TableCell>Patient</TableCell>
              <TableCell>Diagnosis</TableCell>
              <TableCell>Confidence</TableCell>
              <TableCell>Date</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {diagnoses.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  No completed diagnoses available for reporting
                </TableCell>
              </TableRow>
            ) : (
              diagnoses.map((diagnosis) => (
                <TableRow key={diagnosis.id}>
                  <TableCell>{diagnosis.patient?.case_id}</TableCell>
                  <TableCell>
                    {diagnosis.patient?.first_name} {diagnosis.patient?.last_name} ({diagnosis.patient?.age}y {diagnosis.patient?.sex})
                  </TableCell>
                  <TableCell>
                    {diagnosis.primary_diagnosis || 'N/A'}
                  </TableCell>
                  <TableCell>
                    {diagnosis.confidence !== null
                      ? `${(diagnosis.confidence * 100).toFixed(1)}%`
                      : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {new Date(diagnosis.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<PdfIcon />}
                        onClick={() => {
                          setReportFormat('pdf');
                          openGenerateDialog(diagnosis);
                        }}
                      >
                        PDF
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<TextIcon />}
                        onClick={() => {
                          setReportFormat('txt');
                          openGenerateDialog(diagnosis);
                        }}
                      >
                        TXT
                      </Button>
                    </Box>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Generate Report Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Generate Diagnostic Report</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Case ID"
              value={selectedDiagnosis?.patient?.case_id || ''}
              disabled
              sx={{ mb: 3 }}
            />

            <Typography variant="subtitle2" gutterBottom>
              Report Format
            </Typography>
            <ToggleButtonGroup
              color="primary"
              value={reportFormat}
              exclusive
              onChange={handleFormatChange}
              sx={{ mb: 3 }}
            >
              <ToggleButton value="pdf">
                <PdfIcon sx={{ mr: 1 }} />
                PDF
              </ToggleButton>
              <ToggleButton value="txt">
                <TextIcon sx={{ mr: 1 }} />
                Text
              </ToggleButton>
            </ToggleButtonGroup>

            {reportFormat === 'pdf' && (
              <FormControlLabel
                control={
                  <Checkbox
                    checked={includeImages}
                    onChange={(e) => setIncludeImages(e.target.checked)}
                  />
                }
                label="Include images in report"
              />
            )}

            <Alert severity="info" sx={{ mt: 2 }}>
              {reportFormat === 'pdf'
                ? 'PDF reports include formatted layout with optional images. Best for sharing with doctors.'
                : 'Text reports are lightweight and easy to share via SMS/WhatsApp. Best for quick sharing.'}
            </Alert>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleGenerateReport}
            variant="contained"
            startIcon={generating ? null : <DownloadIcon />}
            disabled={generating}
          >
            {generating ? 'Generating...' : `Download ${reportFormat.toUpperCase()}`}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Reports;
