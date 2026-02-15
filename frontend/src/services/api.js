import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Patient APIs
export const createPatient = (patientData) => api.post('/patients/', patientData);
export const getPatients = (skip = 0, limit = 100) =>
  api.get(`/patients/?skip=${skip}&limit=${limit}`);
export const getPatient = (patientId) => api.get(`/patients/${patientId}`);
export const uploadClinicalHistoryPDF = (patientId, file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post(`/patients/${patientId}/upload-history-pdf`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Image Upload APIs
export const uploadImage = (patientId, imageType, file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post(`/upload/${patientId}/${imageType}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Diagnosis APIs
export const createDiagnosis = (diagnosisData) => api.post('/diagnosis/', diagnosisData);
export const getDiagnoses = (skip = 0, limit = 50) =>
  api.get(`/diagnosis/?skip=${skip}&limit=${limit}`);
export const getDiagnosis = (diagnosisId) => api.get(`/diagnosis/${diagnosisId}`);
export const getPatientDiagnoses = (patientId) => api.get(`/diagnosis/patient/${patientId}`);
export const retryDiagnosis = (diagnosisId) => api.post(`/diagnosis/${diagnosisId}/retry`);

// Report APIs
export const generateReport = (diagnosisId, includeImages = true, format = 'pdf') =>
  api.post(`/reports/generate?format=${format}`, { diagnosis_id: diagnosisId, include_images: includeImages });
export const downloadReport = (filename) => api.get(`/reports/download/${filename}`, {
  responseType: 'blob',
});

// Stats API
export const getStats = () => api.get('/stats');

// Health Check
export const healthCheck = () => axios.get(`${API_BASE_URL.replace('/api', '')}/health`);
export const healthCheckDetailed = () => axios.get(`${API_BASE_URL.replace('/api', '')}/health/detailed`);

export default api;
