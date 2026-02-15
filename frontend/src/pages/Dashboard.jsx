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
  Chip,
  Divider,
} from '@mui/material';
import {
  People as PeopleIcon,
  Assessment as AssessmentIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  SmartToy as SmartToyIcon,
  Circle as CircleIcon,
} from '@mui/icons-material';
import { getStats, healthCheckDetailed } from '../services/api';

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

function ModelCard({ modelKey, model }) {
  const statusColor = model.loaded ? '#4caf50' : '#f44336';
  const bgColor = model.loaded ? 'rgba(76, 175, 80, 0.06)' : 'rgba(244, 67, 54, 0.06)';

  return (
    <Card
      sx={{
        height: '100%',
        borderLeft: `4px solid ${statusColor}`,
        backgroundColor: bgColor,
      }}
    >
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {model.name}
          </Typography>
          <Chip
            label={model.loaded ? 'LOADED' : 'NOT LOADED'}
            size="small"
            sx={{
              backgroundColor: statusColor,
              color: 'white',
              fontWeight: 700,
              fontSize: '0.7rem',
            }}
          />
        </Box>
        <Typography variant="body2" color="textSecondary">
          {model.description}
        </Typography>
      </CardContent>
    </Card>
  );
}

/* ── Agent tree helpers ── */
const agentMeta = {
  orchestrator: { icon: '🎯', name: 'Orchestrator Agent' },
  triage: { icon: '🚦', name: 'Triage Agent' },
  radiologist: { icon: '🩻', name: 'Radiologist Agent' },
  pathologist: { icon: '🔬', name: 'Pathologist Agent' },
  clinical_context: { icon: '📋', name: 'Clinical Context Agent' },
};

const statusColor = {
  ready: '#4caf50',
  degraded: '#ff9800',
  offline: '#f44336',
};

function AgentTreeNode({ agentKey, agent, isLast }) {
  const meta = agentMeta[agentKey] || { icon: '🤖', name: agentKey };
  const color = statusColor[agent.status] || statusColor.offline;

  return (
    <Box sx={{ display: 'flex', position: 'relative', ml: 3, mb: isLast ? 0 : 0.5 }}>
      {/* connector lines */}
      <Box sx={{
        position: 'absolute',
        left: -20,
        top: 0,
        bottom: isLast ? '50%' : 0,
        width: 2,
        bgcolor: 'divider',
      }} />
      <Box sx={{
        position: 'absolute',
        left: -20,
        top: '50%',
        width: 20,
        height: 2,
        bgcolor: 'divider',
      }} />

      {/* dot */}
      <Box sx={{
        position: 'absolute',
        left: -4,
        top: '50%',
        transform: 'translateY(-50%)',
        width: 10,
        height: 10,
        borderRadius: '50%',
        bgcolor: color,
        border: '2px solid white',
        zIndex: 1,
        boxShadow: `0 0 0 2px ${color}40`,
      }} />

      {/* content */}
      <Box sx={{
        ml: 2,
        py: 1,
        px: 2,
        borderRadius: 2,
        bgcolor: `${color}08`,
        borderLeft: `3px solid ${color}`,
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 1,
      }}>
        <Box display="flex" alignItems="center" gap={1}>
          <Typography sx={{ fontSize: '1.2rem', lineHeight: 1 }}>{meta.icon}</Typography>
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.2 }}>
              {meta.name}
            </Typography>
            <Typography variant="caption" color="textSecondary">
              {agent.description}
            </Typography>
          </Box>
        </Box>
        <Box display="flex" alignItems="center" gap={0.5}>
          {agent.depends_on?.length > 0 && agent.depends_on.map(d => (
            <Chip key={d} label={d} variant="outlined" size="small"
              sx={{ fontSize: '0.6rem', height: 18, '& .MuiChip-label': { px: 0.8 } }} />
          ))}
          <Chip
            icon={<CircleIcon sx={{ fontSize: '8px !important', color: 'white !important' }} />}
            label={agent.status.toUpperCase()}
            size="small"
            sx={{
              bgcolor: color,
              color: 'white',
              fontWeight: 700,
              fontSize: '0.6rem',
              height: 22,
            }}
          />
        </Box>
      </Box>
    </Box>
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
          healthCheckDetailed(),
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

  // Order agents as a tree: orchestrator root → children
  const agentOrder = ['orchestrator', 'triage', 'radiologist', 'pathologist', 'clinical_context'];

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

      {/* ── Statistics (TOP) ── */}
      <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <AssessmentIcon color="primary" />
        Statistics
      </Typography>
      {stats && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard title="Total Patients" value={stats.total_patients} icon={PeopleIcon} color="#1976d2" />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard title="Total Diagnoses" value={stats.total_diagnoses} icon={AssessmentIcon} color="#388e3c" />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard title="Completed" value={stats.completed_diagnoses} icon={CheckCircleIcon} color="#4caf50" />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard title="TB Detected" value={stats.tb_detected} icon={WarningIcon} color="#f44336" />
          </Grid>
        </Grid>
      )}

      <Divider sx={{ my: 2 }} />

      {/* ── ML Models ── */}
      {health?.models && (
        <>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            🧠 ML Models
          </Typography>
          <Grid container spacing={2} sx={{ mb: 4 }}>
            {Object.entries(health.models).map(([key, model]) => (
              <Grid item xs={12} sm={6} md={4} key={key}>
                <ModelCard modelKey={key} model={model} />
              </Grid>
            ))}
          </Grid>
        </>
      )}

      <Divider sx={{ my: 2 }} />

      {/* ── AI Agents (Tree) ── */}
      {health?.agents && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SmartToyIcon color="primary" />
            AI Agent Pipeline
          </Typography>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            The Orchestrator coordinates all specialist agents in a multi-step diagnostic pipeline.
          </Typography>

          {/* Root: Orchestrator */}
          {health.agents.orchestrator && (
            <Box sx={{ mb: 1 }}>
              <Box sx={{
                py: 1.2, px: 2, borderRadius: 2,
                bgcolor: `${statusColor[health.agents.orchestrator.status] || '#4caf50'}10`,
                border: `2px solid ${statusColor[health.agents.orchestrator.status] || '#4caf50'}`,
                display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1,
              }}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography sx={{ fontSize: '1.4rem', lineHeight: 1 }}>🎯</Typography>
                  <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
                      Orchestrator Agent
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {health.agents.orchestrator.description}
                    </Typography>
                  </Box>
                </Box>
                <Chip
                  icon={<CircleIcon sx={{ fontSize: '8px !important', color: 'white !important' }} />}
                  label={health.agents.orchestrator.status.toUpperCase()}
                  size="small"
                  sx={{
                    bgcolor: statusColor[health.agents.orchestrator.status] || '#4caf50',
                    color: 'white', fontWeight: 700, fontSize: '0.7rem',
                  }}
                />
              </Box>

              {/* Vertical trunk line */}
              <Box sx={{ position: 'relative', pl: 3 }}>
                {/* trunk */}
                <Box sx={{
                  position: 'absolute',
                  left: 24,
                  top: 0,
                  bottom: 0,
                  width: 2,
                  bgcolor: 'divider',
                }} />

                {/* Child agents */}
                {agentOrder.filter(k => k !== 'orchestrator').map((key, idx, arr) => (
                  health.agents[key] && (
                    <AgentTreeNode
                      key={key}
                      agentKey={key}
                      agent={health.agents[key]}
                      isLast={idx === arr.length - 1}
                    />
                  )
                ))}
              </Box>
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
}

export default Dashboard;
