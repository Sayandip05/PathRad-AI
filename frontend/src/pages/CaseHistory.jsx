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
} from '@mui/material';
import { Visibility as VisibilityIcon } from '@mui/icons-material';
import { getDiagnoses, getDiagnosis, retryDiagnosis } from '../services/api';

function CaseHistory() {
  const [diagnoses, setDiagnoses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDiagnosis, setSelectedDiagnosis] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    fetchDiagnoses();
  }, []);

  const fetchDiagnoses = async () => {
    try {
      setLoading(true);
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
      const response = await getDiagnosis(diagnosisId);
      setSelectedDiagnosis(response.data);
      setDialogOpen(true);
    } catch (err) {
      console.error(err);
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
      case 'completed':
        return 'success';
      case 'pending':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      default:
        return 'default';
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
        Case History
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

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
                <TableCell>{diagnosis.patient?.case_id}</TableCell>
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
                  {diagnosis.urgency_level && (
                    <Chip
                      label={diagnosis.urgency_level}
                      color={getUrgencyColor(diagnosis.urgency_level)}
                      size="small"
                    />
                  )}
                </TableCell>
                <TableCell>
                  {diagnosis.tb_probability !== null
                    ? `${(diagnosis.tb_probability * 100).toFixed(1)}%`
                    : 'N/A'}
                </TableCell>
                <TableCell>
                  <Button
                    size="small"
                    startIcon={<VisibilityIcon />}
                    onClick={() => handleViewDetails(diagnosis.id)}
                  >
                    View
                  </Button>
                  {diagnosis.status === 'error' && (
                    <Button
                      size="small"
                      onClick={() => handleRetry(diagnosis.id)}
                      sx={{ ml: 1 }}
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

      {/* Detail Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Diagnosis Details - {selectedDiagnosis?.patient?.case_id}
        </DialogTitle>
        <DialogContent>
          {detailLoading ? (
            <LinearProgress />
          ) : selectedDiagnosis ? (
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Patient Information
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
                          primary="Age"
                          secondary={selectedDiagnosis.patient?.age}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Sex"
                          secondary={selectedDiagnosis.patient?.sex}
                        />
                      </ListItem>
                      {selectedDiagnosis.patient?.location && (
                        <ListItem>
                          <ListItemText
                            primary="Location"
                            secondary={selectedDiagnosis.patient?.location}
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
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Diagnosis Results
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText
                          primary="Primary Diagnosis"
                          secondary={selectedDiagnosis.primary_diagnosis || 'Pending'}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Confidence"
                          secondary={selectedDiagnosis.confidence !== null
                            ? `${(selectedDiagnosis.confidence * 100).toFixed(1)}%`
                            : 'N/A'}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="TB Probability"
                          secondary={selectedDiagnosis.tb_probability !== null
                            ? `${(selectedDiagnosis.tb_probability * 100).toFixed(1)}%`
                            : 'N/A'}
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
              {selectedDiagnosis.findings && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Key Findings
                      </Typography>
                      <List>
                        {selectedDiagnosis.findings.map((finding, idx) => (
                          <ListItem key={idx}>
                            <ListItemText primary={finding} />
                          </ListItem>
                        ))}
                      </List>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default CaseHistory;
