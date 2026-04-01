import React, { useState, useRef, useEffect } from 'react';
import {
  Upload, Leaf, AlertCircle, CheckCircle2, CloudFog, MapPin, Activity, 
  ChevronRight, Thermometer, ShieldCheck, Clock, Sprout, Loader2, Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASENAME = import.meta.env.VITE_API_URL || '';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [crop, setCrop] = useState('');
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  
  const [isRecommending, setIsRecommending] = useState(false);
  const [recommendationData, setRecommendationData] = useState(null);
  
  const [error, setError] = useState('');
  const [gpsLoading, setGpsLoading] = useState(false);
  const [gpsFetched, setGpsFetched] = useState(false);
  const [supportedCrops, setSupportedCrops] = useState(['Apple', 'Grape', 'Tomato']);
  const fileInputRef = useRef(null);

  // Auto-detect GPS on mount
  useEffect(() => {
    fetch(`${API_BASENAME}/api/health`)
      .then(res => res.json())
      .catch(err => console.log('Backend not connected yet:', err));
      
    fetch(`${API_BASENAME}/api/classes`)
      .then(res => res.json())
      .then(data => {
        if (data.crops && data.crops.length > 0) {
          setSupportedCrops(data.crops);
        }
      })
      .catch(() => {});

    // Silently attempt GPS on load
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLatitude(pos.coords.latitude.toFixed(6));
          setLongitude(pos.coords.longitude.toFixed(6));
          setGpsFetched(true);
        },
        () => {} // silently ignore if denied
      );
    }
  }, []);

  const fetchGPS = () => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by your browser.');
      return;
    }
    setGpsLoading(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLatitude(pos.coords.latitude.toFixed(6));
        setLongitude(pos.coords.longitude.toFixed(6));
        setGpsFetched(true);
        setGpsLoading(false);
      },
      () => {
        setError('GPS access denied. Please enable location permission.');
        setGpsLoading(false);
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      const objectUrl = URL.createObjectURL(selected);
      setPreview(objectUrl);
      
      // Reset state on new image
      setPredictionData(null);
      setRecommendationData(null);
      setError('');
    }
  };

  const clearFile = (e) => {
    e.stopPropagation();
    setFile(null);
    setPreview(null);
    setPredictionData(null);
    setRecommendationData(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('image/')) {
        setFile(droppedFile);
        setPreview(URL.createObjectURL(droppedFile));
        setPredictionData(null);
        setRecommendationData(null);
        setError('');
      } else {
        setError('Please drop a valid image file (JPEG, PNG).');
      }
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError('Please select an image first.');
      return;
    }

    setIsPredicting(true);
    setError('');
    setPredictionData(null);
    setRecommendationData(null);

    const formData = new FormData();
    formData.append('file', file);
    if (crop) formData.append('crop', crop);
    if (latitude) formData.append('latitude', latitude);
    if (longitude) formData.append('longitude', longitude);

    try {
      const response = await fetch(`${API_BASENAME}/api/predict`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setPredictionData(data);
      
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      // Automatically trigger recommendation if prediction succeeded and confidence is good
      if (data.success && data.prediction && !data.needs_recapture) {
        handleRecommend(data.prediction.crop, data.prediction.disease, data.prediction.confidence);
      }
      
    } catch (err) {
      setError(err.message || 'Failed to connect to the prediction server. Is it running?');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleRecommend = async (predictedCrop, predictedDisease, confidence) => {
    setIsRecommending(true);
    try {
      const response = await fetch(`${API_BASENAME}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop: predictedCrop,
          disease: predictedDisease,
          confidence: confidence
        }),
      });
      
      const data = await response.json();
      if (data.success) {
        setRecommendationData(data.recommendation);
      } else {
        console.error('Recommendation failed:', data.error);
        setError(`LLM Recommendation Failed: ${data.error}`);
      }
    } catch (err) {
      console.error('Failed to fetch recommendation:', err);
    } finally {
      setIsRecommending(false);
    }
  };

  // Helper for severity color
  const getSeverityClass = (sev) => {
    if (!sev) return 'medium';
    const s = sev.toLowerCase();
    if (s.includes('low')) return 'low';
    if (s.includes('high') || s.includes('severe')) return 'high';
    return 'medium';
  };

  // Treatment Tabs Component
  const TreatmentTabs = ({ organic = [], chemical = [] }) => {
    const [activeTab, setActiveTab] = useState(organic.length > 0 ? 'organic' : 'chemical');
    
    if (organic.length === 0 && chemical.length === 0) return null;

    return (
      <div className="treatment-container">
        <div className="treatment-tabs">
          {organic.length > 0 && (
            <button 
              className={`treatment-tab ${activeTab === 'organic' ? 'active' : ''}`}
              onClick={() => setActiveTab('organic')}
            >
              Organic Treatment
            </button>
          )}
          {chemical.length > 0 && (
            <button 
              className={`treatment-tab ${activeTab === 'chemical' ? 'active' : ''}`}
              onClick={() => setActiveTab('chemical')}
            >
              Chemical Treatment
            </button>
          )}
        </div>
        
        <div className="treatment-content">
          <ul className="rec-list numbered">
            {activeTab === 'organic' 
              ? organic.map((txt, i) => <li key={`org-${i}`}>{txt}</li>)
              : chemical.map((txt, i) => <li key={`chem-${i}`}>{txt}</li>)
            }
          </ul>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <div className="app-content">
        
        {/* Header */}
        <header className="header slide-up">
          <div className="logo">
            <div className="logo-icon">
              <Leaf color="white" size={24} />
            </div>
            <div className="logo-text">
              <span className="logo-title">AgriVisionAI</span>
              <span className="logo-subtitle">Crop Intelligence</span>
            </div>
          </div>
          <div className="header-status">
            <div className="status-badge">
              <div className="status-dot"></div>
              <span>System Online</span>
            </div>
          </div>
        </header>

        {/* Hero */}
        <section className="hero slide-up" style={{ animationDelay: '0.1s' }}>
          <h1>Detect. Diagnose. <span className="highlight">Recover.</span></h1>
          <p>End-to-end AI crop disease intelligence. Upload a leaf image below to receive an instant diagnosis and context-aware LLM recovery plan.</p>
        </section>

        {/* Main Grid */}
        <div className="main-grid">
          
          {/* Left Column: Input Panel */}
          <motion.div 
            className="card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="card-title">
              <Upload size={18} className="icon" /> Input Parameters
            </div>
            
            <div className="form-group">
              <label className="form-label">Target Crop (Optional)</label>
              <select 
                className="form-select" 
                value={crop} 
                onChange={(e) => setCrop(e.target.value)}
              >
                <option value="">Auto-detect crop</option>
                {supportedCrops.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <label className="form-label" style={{ margin: 0 }}>GPS Location</label>
                <button
                  onClick={fetchGPS}
                  disabled={gpsLoading}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '5px',
                    background: gpsFetched ? 'rgba(34,197,94,0.15)' : 'rgba(34,197,94,0.08)',
                    border: `1px solid ${gpsFetched ? 'rgba(34,197,94,0.5)' : 'rgba(34,197,94,0.2)'}`,
                    color: 'var(--accent-green)', borderRadius: '6px',
                    padding: '4px 10px', fontSize: '12px', cursor: 'pointer', fontWeight: 600
                  }}
                >
                  <MapPin size={12} />
                  {gpsLoading ? 'Locating...' : gpsFetched ? '✓ Located' : 'Use My Location'}
                </button>
              </div>
              <div className="form-row">
                <input
                  type="number"
                  step="any"
                  className="form-input"
                  placeholder="Latitude"
                  value={latitude}
                  onChange={(e) => setLatitude(e.target.value)}
                />
                <input
                  type="number"
                  step="any"
                  className="form-input"
                  placeholder="Longitude"
                  value={longitude}
                  onChange={(e) => setLongitude(e.target.value)}
                />
              </div>
              {gpsFetched && (
                <div style={{ fontSize: '11px', color: 'var(--accent-green)', marginTop: '4px', opacity: 0.7 }}>
                  📍 {latitude}, {longitude}
                </div>
              )}
            </div>

            <div className="form-group" style={{ marginTop: '24px' }}>
              <label className="form-label">Leaf Image Capture</label>
              
              <div 
                className={`upload-zone ${preview ? 'has-image' : ''}`}
                onClick={() => !preview && fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {!preview ? (
                  <>
                    <CloudFog className="upload-icon" color="var(--accent-green)" />
                    <div className="upload-title">Drag & drop image here</div>
                    <div className="upload-subtitle">
                      or <span className="upload-browse">browse files</span> (JPEG, PNG)
                    </div>
                  </>
                ) : (
                  <div className="preview-container">
                    <img src={preview} alt="Crop preview" className="preview-image" />
                    <div className="preview-overlay">
                      <button className="preview-btn" onClick={clearFile} title="Remove image">
                        ✕
                      </button>
                    </div>
                  </div>
                )}
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  accept="image/jpeg,image/png,image/webp" 
                  style={{ display: 'none' }} 
                />
              </div>
            </div>

            {error && (
              <div className="guardrail-item error" style={{ marginTop: '16px' }}>
                <AlertCircle size={16} className="guardrail-icon" />
                <span>{error}</span>
              </div>
            )}

            <button 
              className="btn btn-primary" 
              style={{ marginTop: '24px' }}
              onClick={handlePredict}
              disabled={!file || isPredicting}
            >
              {isPredicting ? (
                <><div className="spinner"></div> Running Inference...</>
              ) : (
                <><Activity size={18} /> Analyze Crop Intelligence</>
              )}
            </button>
          </motion.div>

          {/* Right Column: Prediction Results */}
          <motion.div 
            className="card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="card-title">
              <ShieldCheck size={18} className="icon" /> Analysis Results
            </div>
            
            {/* Empty State */}
            {!predictionData && !isPredicting && (
              <div className="empty-state">
                <Leaf className="empty-state-icon" />
                <div className="empty-state-text">
                  Awaiting image input.<br/>Upload a clear picture of a leaf to begin analysis.
                </div>
              </div>
            )}
            
            {/* Loading State */}
            {isPredicting && (
              <div className="empty-state">
                <div className="spinner" style={{ margin: '0 auto 16px', width: '30px', height: '30px', borderColor: 'var(--accent-green)', borderTopColor: 'transparent' }}></div>
                <div className="empty-state-text">Processing image...<br/>Running deep learning model</div>
              </div>
            )}

            <AnimatePresence>
              {predictionData && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="result-section"
                >
                  {/* Warnings Summary */}
                  {predictionData.warnings && predictionData.warnings.length > 0 && (
                    <div className="guardrail-item warning" style={{ marginBottom: '16px', display: 'block' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontWeight: '600' }}>
                        <AlertCircle size={16} /> Analysis Warnings
                      </div>
                      <ul style={{ paddingLeft: '24px', margin: 0, fontSize: '13px' }}>
                        {predictionData.warnings.map((w, i) => <li key={i}>{w}</li>)}
                      </ul>
                      {predictionData.needs_recapture && (
                        <div style={{ marginTop: '8px', fontWeight: '600', color: 'var(--color-error)' }}>
                          Recapture is recommended for accurate results.
                        </div>
                      )}
                    </div>
                  )}

                  {/* Terminal Hacker Style Output */}
                  {predictionData.prediction && recommendationData && (
                    <div className="terminal-block">
                      <div className="term-line"><span className="term-key">Crop:</span> <span className="term-val">{predictionData.prediction.crop}</span></div>
                      <div className="term-line"><span className="term-key">Disease:</span> <span className="term-val">{predictionData.prediction.disease}</span></div>
                      <div className="term-line"><span className="term-key">Confidence:</span> <span className="term-val">{(predictionData.prediction.confidence * 100).toFixed(0)}%</span></div>
                      <div className="term-line"><span className="term-key">Severity:</span> <span className="term-val">{recommendationData.severity}</span></div>
                      <div className="term-line"><span className="term-key">Location:</span> <span className="term-val">{recommendationData.location_context}</span></div>
                      <div className="term-line"><span className="term-key">Time Context:</span> <span className="term-val">{recommendationData.time_context}</span></div>
                      <br/>
                      <div className="term-divider">--- AI Recommendation ---</div>
                      <br/>
                      {recommendationData.recommendations.map((rec, i) => (
                        <div key={i} className="term-list-item">{rec}</div>
                      ))}
                      <br/>
                      <div className="term-line"><span className="term-key-alt">Est. Recovery:</span> <span className="term-val">{recommendationData.recovery_time}</span></div>
                      <div className="term-line"><span className="term-key-alt">Preventive Note:</span> <span className="term-val">{recommendationData.preventive_note}</span></div>
                    </div>
                  )}

                  {/* Fallback while recommending */}
                  {predictionData.prediction && !recommendationData && (
                    <div className="prediction-main">
                      <div className="prediction-header">
                        <div>
                          <div className="prediction-disease">{predictionData.prediction.disease}</div>
                          <div className="prediction-crop">{predictionData.prediction.crop} Pipeline</div>
                        </div>
                      </div>
                      
                      <div className="confidence-section">
                        <div className="confidence-label">
                          <span>AI Confidence Score</span>
                          <span className="confidence-value">
                            {(predictionData.prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="confidence-bar">
                          <div 
                            className={`confidence-fill ${predictionData.prediction.confidence < 0.6 ? 'low' : ''}`}
                            style={{ width: `${predictionData.prediction.confidence * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Guardrails List Component */}
                  {predictionData.guardrails && predictionData.guardrails.length > 0 && (
                     <div className="guardrails">
                       <div className="guardrails-title">
                         <ShieldCheck size={14} /> Quality Gates
                       </div>
                       {predictionData.guardrails.map((gr, idx) => {
                         const Icon = gr.passed ? CheckCircle2 : (gr.severity === 'error' ? AlertCircle : Info);
                         const statusClass = gr.passed ? 'passed' : gr.severity;
                         
                         // Map specific checks to cool icons dynamically if we wanted, 
                         // but for speed we just output them clean
                         return (
                           <div key={idx} className={`guardrail-item ${statusClass}`}>
                             <Icon className="guardrail-icon" />
                             <span>{gr.message}</span>
                           </div>
                         );
                       })}
                     </div>
                  )}
                  
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>

        {/* Bottom Full-Width Section: LLM Recommendation Loading Only */}
        <AnimatePresence>
          {isRecommending && (
            <motion.div 
              className="recommendation-panel"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="card" style={{ borderTop: '4px solid var(--accent-green)' }}>
                <div className="card-title">
                  <Sprout size={20} className="icon" /> Groq LLM Recovery Recommendation
                </div>
                <div className="empty-state" style={{ padding: '20px' }}>
                  <div className="spinner" style={{ margin: '0 auto 16px', width: '24px', height: '24px', borderColor: 'var(--accent-green)', borderTopColor: 'transparent' }}></div>
                  <div className="empty-state-text">Consulting Ag-LLM...<br/>Generating context-aware recovery plan</div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="footer slide-up" style={{ animationDelay: '0.4s' }}>
          <div className="footer-text">
            Matrix Fusion 4.0 Hackathon • Powered by <a href="https://kokos.ai">Kokos.ai</a> & Groq LLM
          </div>
        </footer>

      </div>
    </div>
  );
}

export default App;
