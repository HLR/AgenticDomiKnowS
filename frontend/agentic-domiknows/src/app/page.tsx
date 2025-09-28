'use client';

import { useState } from 'react';
import LandingPage from '@/components/LandingPage';
import MainApp from '@/components/MainApp';

export default function App() {
  const [showMainApp, setShowMainApp] = useState(false);

  const handleChooseNewApp = () => {
    setShowMainApp(true);
  };

  if (showMainApp) {
    return <MainApp />;
  }

  return <LandingPage onChooseNewApp={handleChooseNewApp} />;
}