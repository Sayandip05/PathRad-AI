import React, { useState } from 'react';
import {
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Box,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Paper,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  CloudUpload as CloudUploadIcon,
  Description as DescriptionIcon,
} from '@mui/icons-material';
import { createPatient, uploadImage, createDiagnosis, uploadClinicalHistoryPDF } from '../services/api';
import VoiceInput from '../components/VoiceInput';

const steps = ['Patient Information', 'Upload Images', 'Review & Submit'];

function NewCase() {
  const [activeStep, setActiveStep] = useState(0);
  const [patientData, setPatientData] = useState({
    first_name: '',
    last_name: '',
    age: '',
    sex: '',
    location: '',
    chief_complaint: '',
    clinical_history: '',
  });
  const [clinicalHistoryPdf, setClinicalHistoryPdf] = useState(null);
  const [xrayFile, setXrayFile] = useState(null);
  const [microscopyFile, setMicroscopyFile] = useState(null);
  const [patientId, setPatientId] = useState(null);
  const [xrayImageId, setXrayImageId] = useState(null);
  const [microscopyImageId, setMicroscopyImageId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [diagnosisId, setDiagnosisId] = useState(null);

  const handlePatientChange = (field) => (event) => {
    setPatientData({ ...patientData, [field]: event.target.value });
  };

  const handleFileChange = (type) => (event) => {
    if (type === 'xray') {
      setXrayFile(event.target.files[0]);
    } else {
      setMicroscopyFile(event.target.files[0]);
    }
  };

  const handleNext = async () => {
    setError(null);
    setLoading(true);

    try {
      if (activeStep === 0) {
        // Create patient
        const response = await createPatient(patientData);
        const newPatientId = response.data.id;
        setPatientId(newPatientId);

        // Upload clinical history PDF if provided
        if (clinicalHistoryPdf) {
          await uploadClinicalHistoryPDF(newPatientId, clinicalHistoryPdf);
        }
      } else if (activeStep === 1) {
        // Upload images
        if (xrayFile) {
          const xrayResponse = await uploadImage(patientId, 'xray', xrayFile);
          setXrayImageId(xrayResponse.data.image_id);
        }
        if (microscopyFile) {
          const microResponse = await uploadImage(patientId, 'microscopy', microscopyFile);
          setMicroscopyImageId(microResponse.data.image_id);
        }
      } else if (activeStep === 2) {
        // Create diagnosis
        const diagnosisResponse = await createDiagnosis({
          patient_id: patientId,
          xray_image_id: xrayImageId,
          microscopy_image_id: microscopyImageId,
        });
        setDiagnosisId(diagnosisResponse.data.id);
        setSuccess(true);
      }

      setActiveStep((prevStep) => prevStep + 1);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                required
                fullWidth
                label="First Name"
                value={patientData.first_name}
                onChange={handlePatientChange('first_name')}
                placeholder="e.g., John"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                required
                fullWidth
                label="Last Name"
                value={patientData.last_name}
                onChange={handlePatientChange('last_name')}
                placeholder="e.g., Doe"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                required
                fullWidth
                label="Age"
                type="number"
                value={patientData.age}
                onChange={handlePatientChange('age')}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Sex</InputLabel>
                <Select
                  value={patientData.sex}
                  onChange={handlePatientChange('sex')}
                  label="Sex"
                >
                  <MenuItem value="male">Male</MenuItem>
                  <MenuItem value="female">Female</MenuItem>
                  <MenuItem value="other">Other</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Location"
                value={patientData.location}
                onChange={handlePatientChange('location')}
                placeholder="e.g., Mumbai, India"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Chief Complaint"
                multiline
                rows={2}
                value={patientData.chief_complaint}
                onChange={handlePatientChange('chief_complaint')}
                placeholder="e.g., Cough for 3 weeks, fever, weight loss"
              />
              <Box sx={{ mt: 2 }}>
                <VoiceInput
                  inputType="chief_complaint"
                  onTranscriptionComplete={(text) =>
                    setPatientData({ ...patientData, chief_complaint: text })
                  }
                />
              </Box>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Clinical History"
                multiline
                rows={4}
                value={patientData.clinical_history}
                onChange={handlePatientChange('clinical_history')}
                placeholder="Patient's medical history, risk factors, medications, etc."
              />
              <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<DescriptionIcon />}
                  size="small"
                >
                  Upload History PDF
                  <input
                    type="file"
                    hidden
                    accept=".pdf"
                    onChange={(e) => setClinicalHistoryPdf(e.target.files[0])}
                  />
                </Button>
                {clinicalHistoryPdf && (
                  <Typography variant="body2" color="success.main">
                    📎 {clinicalHistoryPdf.name}
                  </Typography>
                )}
              </Box>
              <Box sx={{ mt: 2 }}>
                <VoiceInput
                  inputType="clinical_history"
                  onTranscriptionComplete={(text) =>
                    setPatientData({ ...patientData, clinical_history: text })
                  }
                />
              </Box>
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Chest X-Ray
                </Typography>
                <input
                  accept="image/*"
                  style={{ display: 'none' }}
                  id="xray-upload"
                  type="file"
                  onChange={handleFileChange('xray')}
                />
                <label htmlFor="xray-upload">
                  <Button
                    variant="contained"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    Upload X-Ray
                  </Button>
                </label>
                {xrayFile && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    Selected: {xrayFile.name}
                  </Alert>
                )}
              </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Microscopy (Optional)
                </Typography>
                <input
                  accept="image/*"
                  style={{ display: 'none' }}
                  id="microscopy-upload"
                  type="file"
                  onChange={handleFileChange('microscopy')}
                />
                <label htmlFor="microscopy-upload">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    Upload Microscopy
                  </Button>
                </label>
                {microscopyFile && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    Selected: {microscopyFile.name}
                  </Alert>
                )}
              </Paper>
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Review Case Information
              </Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary="Patient Name"
                    secondary={`${patientData.first_name} ${patientData.last_name}`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Patient Age"
                    secondary={patientData.age}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Sex"
                    secondary={patientData.sex}
                  />
                </ListItem>
                {patientData.location && (
                  <ListItem>
                    <ListItemText
                      primary="Location"
                      secondary={patientData.location}
                    />
                  </ListItem>
                )}
                <ListItem>
                  <ListItemText
                    primary="Chief Complaint"
                    secondary={patientData.chief_complaint || 'Not provided'}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    {xrayFile ? <CheckCircleIcon color="success" /> : <ErrorIcon color="error" />}
                  </ListItemIcon>
                  <ListItemText
                    primary="X-Ray Image"
                    secondary={xrayFile ? xrayFile.name : 'Not uploaded'}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    {microscopyFile ? <CheckCircleIcon color="success" /> : <CheckCircleIcon color="disabled" />}
                  </ListItemIcon>
                  <ListItemText
                    primary="Microscopy Image"
                    secondary={microscopyFile ? microscopyFile.name : 'Not uploaded (optional)'}
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        );

      default:
        return null;
    }
  };

  if (success) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Alert severity="success" sx={{ mb: 3 }}>
          <Typography variant="h6">Diagnosis Request Submitted Successfully!</Typography>
          <Typography>
            Diagnosis ID: {diagnosisId}
          </Typography>
          <Typography>
            The multi-agent system is now analyzing the case. Check the History page for results.
          </Typography>
        </Alert>
        <Button
          variant="contained"
          onClick={() => window.location.reload()}
          sx={{ mt: 2 }}
        >
          Create Another Case
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        New Diagnostic Case
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Paper sx={{ p: 3, mb: 3 }}>
        {renderStepContent(activeStep)}
      </Paper>

      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
        >
          Back
        </Button>
        <Button
          variant="contained"
          onClick={handleNext}
          disabled={
            loading ||
            (activeStep === 0 && (!patientData.first_name || !patientData.last_name || !patientData.age || !patientData.sex)) ||
            (activeStep === 1 && !xrayFile)
          }
        >
          {loading ? <CircularProgress size={24} /> : activeStep === steps.length - 1 ? 'Submit' : 'Next'}
        </Button>
      </Box>
    </Box>
  );
}

export default NewCase;
