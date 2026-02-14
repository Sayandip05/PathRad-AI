import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Button,
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  PersonAdd as PersonAddIcon,
  Assessment as AssessmentIcon,
  History as HistoryIcon,
} from '@mui/icons-material';

// Pages
import Dashboard from './pages/Dashboard';
import NewCase from './pages/NewCase';
import CaseHistory from './pages/CaseHistory';
import Reports from './pages/Reports';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              PathRad AI
            </Typography>
            <Button color="inherit" component={Link} to="/" startIcon={<DashboardIcon />}>
              Dashboard
            </Button>
            <Button color="inherit" component={Link} to="/new-case" startIcon={<PersonAddIcon />}>
              New Case
            </Button>
            <Button color="inherit" component={Link} to="/history" startIcon={<HistoryIcon />}>
              History
            </Button>
            <Button color="inherit" component={Link} to="/reports" startIcon={<AssessmentIcon />}>
              Reports
            </Button>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new-case" element={<NewCase />} />
            <Route path="/history" element={<CaseHistory />} />
            <Route path="/reports" element={<Reports />} />
          </Routes>
        </Container>

        <Box sx={{ bgcolor: 'background.paper', p: 2, mt: 'auto' }} component="footer">
          <Typography variant="body2" color="text.secondary" align="center">
            PathRad AI - Multi-Agent Diagnostic System © 2024
          </Typography>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
