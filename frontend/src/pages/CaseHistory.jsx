import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  HourglassEmpty as PendingIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { getDiagnoses, getDiagnosis, retryDiagnosis } from '../services/api';

function AgentStatusChip({ label, data, skippedText }) {
  if (!data) {
    return (
      <Chip
        icon={<PendingIcon sx={{ fontSize: 14 }} />}
        label={`${label}: Pending`}
        size="small"
        variant="outlined"
        sx={{ m: 0.3 }}
      />
    );
  }
  if (data.status === 'skipped') {
    return (
      <Chip
        label={`${label}: ${skippedText || 'Skipped'}`}
        size="small"
        variant="outlined"
        color="default"
        sx={{ m: 0.3 }}
      />
    );
  }
  if (data.error) {
    return (
      <Chip
        icon={<CancelIcon sx={{ fontSize: 14 }} />}
        label={`${label}: Error`}
        size="small"
        color="error"
        sx={{ m: 0.3 }}
      />
    );
  }
  return (
    <Chip
      icon={<CheckCircleIcon sx={{ fontSize: 14 }} />}
      label={`${label}: Done`}
      size="small"
      color="success"
      sx={{ m: 0.3 }}
    />
  );
}

function CaseHistory() {
  const [diagnoses, setDiagnoses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDiagnosis, setSelectedDiagnosis] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(null);

  useEffect(() => {
    fetchDiagnoses();
  }, []);

  const fetchDiagnoses = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await getDiagnoses();
      setDiagnoses(response.data);
    } catch (err) {
      setError('Failed to load diagnoses');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = async (diagnosisId) => {
    try {
      setDetailLoading(true);
      setDetailError(null);
      const response = await getDiagnosis(diagnosisId);
      setSelectedDiagnosis(response.data);
      setDialogOpen(true);
    } catch (err) {
      console.error(err);
      setDetailError(`Failed to load diagnosis details: ${err?.response?.data?.detail || err.message}`);
      setDialogOpen(true);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleRetry = async (diagnosisId) => {
    try {
      await retryDiagnosis(diagnosisId);
      fetchDiagnoses();
    } catch (err) {
      console.error(err);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'pending': return 'warning';
      case 'error': return 'error';
      case 'ESCALATED_TO_HUMAN': return 'info';
      default: return 'default';
    }
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      default: return 'default';
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
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h4">
          Case History
        </Typography>
        <Button
          startIcon={<RefreshIcon />}
          onClick={fetchDiagnoses}
          variant="outlined"
          size="small"
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {diagnoses.length === 0 ? (
        <Alert severity="info">
          No diagnoses found. Create a new case to start the diagnostic pipeline.
        </Alert>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Case ID</TableCell>
                <TableCell>Patient</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Urgency</TableCell>
                <TableCell>TB Probability</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {diagnoses.map((diagnosis) => (
                <TableRow key={diagnosis.id}>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                      {diagnosis.patient?.case_id || diagnosis.id.slice(0, 8)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {diagnosis.patient?.first_name} {diagnosis.patient?.last_name} ({diagnosis.patient?.age}y {diagnosis.patient?.sex})
                  </TableCell>
                  <TableCell>
                    {new Date(diagnosis.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={diagnosis.status}
                      color={getStatusColor(diagnosis.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {diagnosis.urgency_level ? (
                      <Chip
                        label={diagnosis.urgency_level}
                        color={getUrgencyColor(diagnosis.urgency_level)}
                        size="small"
                      />
                    ) : (
                      <Typography variant="caption" color="textSecondary">—</Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {diagnosis.tb_probability !== null && diagnosis.tb_probability !== undefined
                      ? `${(diagnosis.tb_probability * 100).toFixed(1)}%`
                      : 'N/A'}
                  </TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      startIcon={<VisibilityIcon />}
                      onClick={() => handleViewDetails(diagnosis.id)}
                      variant="outlined"
                    >
                      View
                    </Button>
                    {(diagnosis.status === 'error' || diagnosis.status === 'ESCALATED_TO_HUMAN') && (
                      <Button
                        size="small"
                        onClick={() => handleRetry(diagnosis.id)}
                        sx={{ ml: 1 }}
                        color="warning"
                        variant="outlined"
                      >
                        Retry
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Detail Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => { setDialogOpen(false); setDetailError(null); }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Diagnosis Details {selectedDiagnosis?.patient?.case_id ? `— ${selectedDiagnosis.patient.case_id}` : ''}
        </DialogTitle>
        <DialogContent>
          {detailLoading ? (
            <LinearProgress />
          ) : detailError ? (
            <Alert severity="error">{detailError}</Alert>
          ) : selectedDiagnosis ? (
            <Grid container spacing={2} sx={{ mt: 0 }}>
              {/* Patient Info */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      👤 Patient Information
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText
                          primary="Name"
                          secondary={`${selectedDiagnosis.patient?.first_name} ${selectedDiagnosis.patient?.last_name}`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Age / Sex"
                          secondary={`${selectedDiagnosis.patient?.age}y ${selectedDiagnosis.patient?.sex}`}
                        />
                      </ListItem>
                      {selectedDiagnosis.patient?.location && (
                        <ListItem>
                          <ListItemText
                            primary="Location"
                            secondary={selectedDiagnosis.patient.location}
                          />
                        </ListItem>
                      )}
                      <ListItem>
                        <ListItemText
                          primary="Chief Complaint"
                          secondary={selectedDiagnosis.patient?.chief_complaint || 'N/A'}
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              {/* Diagnosis Results */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      🩺 Diagnosis Results
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText
                          primary="Status"
                          secondary={
                            <Chip
                              label={selectedDiagnosis.status}
                              color={getStatusColor(selectedDiagnosis.status)}
                              size="small"
                            />
                          }
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Primary Diagnosis"
                          secondary={selectedDiagnosis.primary_diagnosis || 'Pending...'}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Confidence"
                          secondary={selectedDiagnosis.confidence !== null && selectedDiagnosis.confidence !== undefined
                            ? `${(selectedDiagnosis.confidence * 100).toFixed(1)}%`
                            : 'N/A'}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="TB Probability"
                          secondary={selectedDiagnosis.tb_probability !== null && selectedDiagnosis.tb_probability !== undefined
                            ? `${(selectedDiagnosis.tb_probability * 100).toFixed(1)}%`
                            : 'N/A'}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Human Review Required"
                          secondary={selectedDiagnosis.human_review_required ? '⚠️ Yes' : '✅ No'}
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              {/* Agent Pipeline Status */}
              <Grid item xs={12}>
                <Card variant="outlined" sx={{ borderColor: 'primary.main' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      🤖 Agent Pipeline Execution
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={0.5} mb={2}>
                      <AgentStatusChip label="🚦 Triage" data={selectedDiagnosis.triage_result} />
                      <AgentStatusChip label="🩻 Radiologist" data={selectedDiagnosis.radiology_result} />
                      <AgentStatusChip label="🔬 Pathologist" data={selectedDiagnosis.pathology_details} skippedText="No microscopy" />
                      <AgentStatusChip label="📋 Clinical Context" data={selectedDiagnosis.clinical_context} />
                    </Box>

                    <Divider sx={{ my: 1.5 }} />

                    {/* Triage Details */}
                    {selectedDiagnosis.triage_result && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" color="primary">
                          🚦 Triage Agent
                        </Typography>
                        <Typography variant="body2">
                          Urgency: <strong>{selectedDiagnosis.triage_result.urgency_level || 'N/A'}</strong>
                          {' '}(Score: {selectedDiagnosis.triage_result.urgency_score ?? 'N/A'}/10)
                        </Typography>
                        {selectedDiagnosis.triage_result.critical_flags?.length > 0 && (
                          <Typography variant="body2" color="error">
                            ⚠️ Critical Flags: {selectedDiagnosis.triage_result.critical_flags.join(', ')}
                          </Typography>
                        )}
                      </Box>
                    )}

                    {/* Radiology Details */}
                    {selectedDiagnosis.radiology_result && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" color="primary">
                          🩻 Radiologist Agent
                        </Typography>
                        <Typography variant="body2">
                          Diagnosis: <strong>{selectedDiagnosis.radiology_result.primary_diagnosis || 'N/A'}</strong>
                          {' '}(Confidence: {selectedDiagnosis.radiology_result.confidence
                            ? `${(selectedDiagnosis.radiology_result.confidence * 100).toFixed(1)}%`
                            : 'N/A'})
                        </Typography>
                        {selectedDiagnosis.radiology_result.findings?.length > 0 && (
                          <Typography variant="body2">
                            Findings: {selectedDiagnosis.radiology_result.findings.join(', ')}
                          </Typography>
                        )}
                      </Box>
                    )}

                    {/* Pathology Details */}
                    {selectedDiagnosis.pathology_details && selectedDiagnosis.pathology_details.status !== 'skipped' && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" color="primary">
                          🔬 Pathologist Agent
                        </Typography>
                        <Typography variant="body2">
                          Result: <strong>{selectedDiagnosis.pathology_details.result || 'N/A'}</strong>
                          {' '}— {selectedDiagnosis.pathology_details.quantification || ''}
                        </Typography>
                        <Typography variant="body2">
                          Bacilli Count: {selectedDiagnosis.pathology_details.bacilli_count ?? 'N/A'}
                          {' '}| Confidence: {selectedDiagnosis.pathology_details.confidence
                            ? `${(selectedDiagnosis.pathology_details.confidence * 100).toFixed(1)}%`
                            : 'N/A'}
                        </Typography>
                      </Box>
                    )}

                    {/* Clinical Context Details */}
                    {selectedDiagnosis.clinical_context && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" color="primary">
                          📋 Clinical Context Agent
                        </Typography>
                        {selectedDiagnosis.clinical_context.relevant_symptoms?.length > 0 && (
                          <Typography variant="body2">
                            Symptoms: {selectedDiagnosis.clinical_context.relevant_symptoms.join(', ')}
                          </Typography>
                        )}
                        {selectedDiagnosis.clinical_context.risk_factors?.length > 0 && (
                          <Typography variant="body2">
                            Risk Factors: {selectedDiagnosis.clinical_context.risk_factors.join(', ')}
                          </Typography>
                        )}
                        {selectedDiagnosis.clinical_context.risk_scores?.tb_risk !== undefined && (
                          <Typography variant="body2">
                            TB Risk Score: <strong>{selectedDiagnosis.clinical_context.risk_scores.tb_risk}/10</strong>
                          </Typography>
                        )}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Findings */}
              {selectedDiagnosis.findings && selectedDiagnosis.findings.length > 0 && (
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        📋 Key Findings
                      </Typography>
                      <List dense>
                        {selectedDiagnosis.findings.map((finding, idx) => (
                          <ListItem key={idx}>
                            <ListItemText primary={`• ${finding}`} />
                          </ListItem>
                        ))}
                      </List>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Treatment Plan */}
              {selectedDiagnosis.treatment_plan && (
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        💊 Treatment Plan
                      </Typography>
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
                        {selectedDiagnosis.treatment_plan}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setDialogOpen(false); setDetailError(null); }}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default CaseHistory;
