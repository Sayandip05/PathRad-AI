import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  People as PeopleIcon,
  Assessment as AssessmentIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { getStats, healthCheck } from '../services/api';

function StatCard({ title, value, icon: Icon, color }) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" component="div">
              {value}
            </Typography>
          </Box>
          <Box sx={{ color: color }}>
            <Icon fontSize="large" />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [statsRes, healthRes] = await Promise.all([
          getStats(),
          healthCheck(),
        ]);
        setStats(statsRes.data);
        setHealth(healthRes.data);
      } catch (err) {
        setError('Failed to load dashboard data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

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
        Dashboard
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Health Status */}
      {health && (
        <Alert 
          severity={health.models_loaded.medgemma ? "success" : "warning"}
          sx={{ mb: 3 }}
        >
          System Status: {health.status} | 
          Models: MedGemma {health.models_loaded.medgemma ? '✓' : '✗'}, 
          CXR {health.models_loaded.cxr ? '✓' : '✗'}, 
          Path {health.models_loaded.path ? '✓' : '✗'}
        </Alert>
      )}

      {/* Statistics */}
      {stats && (
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Total Patients"
              value={stats.total_patients}
              icon={PeopleIcon}
              color="#1976d2"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Total Diagnoses"
              value={stats.total_diagnoses}
              icon={AssessmentIcon}
              color="#388e3c"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Completed"
              value={stats.completed_diagnoses}
              icon={CheckCircleIcon}
              color="#4caf50"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="TB Detected"
              value={stats.tb_detected}
              icon={WarningIcon}
              color="#f44336"
            />
          </Grid>
        </Grid>
      )}

      {/* Recent Activity */}
      <Paper sx={{ mt: 3, p: 3 }}>
        <Typography variant="h6" gutterBottom>
          System Overview
        </Typography>
        <Typography variant="body1" paragraph>
          PathRad AI is a multi-agent diagnostic system powered by Google ADK and MedGemma models.
          The system uses 5 specialized AI agents to provide comprehensive diagnostic analysis:
        </Typography>
        <Box component="ul" sx={{ pl: 3 }}>
          <li><strong>Triage Agent:</strong> Rapid initial assessment and quality control</li>
          <li><strong>Radiologist Agent:</strong> Chest X-ray analysis using CXR Foundation</li>
          <li><strong>Pathologist Agent:</strong> Microscopy analysis using Path Foundation</li>
          <li><strong>Clinical Context Agent:</strong> Patient history and risk assessment</li>
          <li><strong>Orchestrator Agent:</strong> Master coordinator and report synthesis</li>
        </Box>
      </Paper>
    </Box>
  );
}

export default Dashboard;
