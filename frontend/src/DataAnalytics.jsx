import React, { useState, useMemo, useEffect, useRef } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, CheckCircle, X, Filter, Download, AlertCircle, RotateCw } from 'lucide-react';
import { API_BASE } from './utils/apiConfig';

// Add custom animations via style tag
const animationStyles = `
  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
  
  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(30px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes pulseSlow {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }
  
  @keyframes bounceSlow {
    0%, 100% {
      transform: translateY(0);
    }
    50% {
      transform: translateY(-5px);
    }
  }
  
  @keyframes shake {
    0%, 100% {
      transform: translateX(0);
    }
    25% {
      transform: translateX(-3px);
    }
    75% {
      transform: translateX(3px);
    }
  }
  
  @keyframes countUp {
    from {
      opacity: 0;
      transform: scale(0.8);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }
  
  .animate-fade-in {
    animation: fadeIn 0.6s ease-in-out;
  }
  
  .animate-slide-in-up {
    animation: slideInUp 0.5s ease-out forwards;
    opacity: 0;
  }
  
  .animate-fade-in-up {
    animation: fadeInUp 0.6s ease-out forwards;
    opacity: 0;
  }
  
  .animate-pulse-slow {
    animation: pulseSlow 2s ease-in-out infinite;
  }
  
  .animate-bounce-slow {
    animation: bounceSlow 2s ease-in-out infinite;
  }
  
  .animate-shake {
    animation: shake 0.5s ease-in-out infinite;
  }
  
  .animate-count-up {
    animation: countUp 0.8s ease-out;
  }
`;

// Inject styles
if (typeof document !== 'undefined') {
  const styleId = 'data-analytics-animations';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = animationStyles;
    document.head.appendChild(style);
  }
}

// Custom Pie Chart Component
const PieChart = ({ data, size = 250, colors = [] }) => {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  if (total === 0) {
    return (
      <div className="flex items-center justify-center" style={{ width: size, height: size }}>
        <p className="text-gray-400">No data available</p>
      </div>
    );
  }

  const radius = size / 2 - 10;
  const centerX = size / 2;
  const centerY = size / 2;
  let currentAngle = -90; // Start at top

  const segments = data.map((item, index) => {
    const percentage = (item.value / total) * 100;
    const angle = (item.value / total) * 360;
    const startAngle = currentAngle;
    const endAngle = currentAngle + angle;
    
    const startAngleRad = (startAngle * Math.PI) / 180;
    const endAngleRad = (endAngle * Math.PI) / 180;
    
    const x1 = centerX + radius * Math.cos(startAngleRad);
    const y1 = centerY + radius * Math.sin(startAngleRad);
    const x2 = centerX + radius * Math.cos(endAngleRad);
    const y2 = centerY + radius * Math.sin(endAngleRad);
    
    const largeArcFlag = angle > 180 ? 1 : 0;
    
    const pathData = [
      `M ${centerX} ${centerY}`,
      `L ${x1} ${y1}`,
      `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
      'Z'
    ].join(' ');

    const labelAngle = startAngle + angle / 2;
    const labelRadius = radius * 0.7;
    const labelX = centerX + labelRadius * Math.cos((labelAngle * Math.PI) / 180);
    const labelY = centerY + labelRadius * Math.sin((labelAngle * Math.PI) / 180);

    const color = colors[index % colors.length] || `hsl(${(index * 360) / data.length}, 70%, 50%)`;
    
    currentAngle += angle;

    return {
      pathData,
      color,
      labelX,
      labelY,
      percentage: percentage.toFixed(1),
      label: item.label,
      value: item.value
    };
  });

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {segments.map((segment, index) => (
          <g key={index}>
            <path
              d={segment.pathData}
              fill={segment.color}
              stroke="#fff"
              strokeWidth="2"
              className="hover:opacity-80 transition-opacity cursor-pointer"
            >
              <title>{segment.label}: {segment.value} ({segment.percentage}%)</title>
            </path>
            {segment.percentage > 5 && (
              <text
                x={segment.labelX}
                y={segment.labelY}
                fill="#fff"
                fontSize="12"
                fontWeight="bold"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {segment.percentage}%
              </text>
            )}
          </g>
        ))}
      </svg>
      <div className="mt-4 space-y-2 w-full max-w-xs">
        {data.map((item, index) => {
          const color = colors[index % colors.length] || `hsl(${(index * 360) / data.length}, 70%, 50%)`;
          return (
            <div key={index} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <div 
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: color }}
                />
                <span className="font-medium text-gray-700">{item.label}</span>
              </div>
              <span className="font-bold text-gray-900">{item.value}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Custom Bar Chart Component - Modern Design
const BarChart = ({ data, maxValue, height = 300, colors = ['#3b82f6'] }) => {
  const maxBarValue = maxValue || Math.max(...data.map(d => d.value), 1);
  const padding = { left: 70, right: 30, top: 50, bottom: 80 };
  const chartHeight = height - padding.top - padding.bottom;
  const chartWidth = 900;
  const barSpacing = 25;
  const availableWidth = chartWidth - padding.left - padding.right;
  const barWidth = Math.min((availableWidth - (barSpacing * (data.length - 1))) / data.length, 150);
  const barBottom = padding.top + chartHeight; // Bottom position of bars

  // Calculate grid intervals dynamically
  const gridIntervals = [];
  if (maxBarValue > 0) {
    const intervals = [0, 25, 50, 75, 100];
    intervals.forEach(percent => {
      if ((maxBarValue * percent / 100) <= maxBarValue) {
        gridIntervals.push(percent);
      }
    });
  }

  return (
    <div className="w-full overflow-x-auto">
      <svg width="100%" height={height} viewBox={`0 0 ${chartWidth} ${height}`} preserveAspectRatio="xMidYMid meet" className="max-w-full">
        {/* Background */}
        <rect x={0} y={0} width={chartWidth} height={height} fill="#fafafa" />
        
        {/* Grid lines with better styling */}
        {gridIntervals.map((percent) => {
          const yPos = padding.top + chartHeight - (chartHeight * percent / 100);
          const value = Math.round((maxBarValue * percent) / 100);
          return (
            <g key={percent}>
              <line
                x1={padding.left}
                y1={yPos}
                x2={chartWidth - padding.right}
                y2={yPos}
                stroke={percent === 0 ? "#9ca3af" : "#e5e7eb"}
                strokeWidth={percent === 0 ? 2 : 1}
                strokeDasharray={percent === 0 ? "0" : "5 5"}
                opacity={0.6}
              />
              <text
                x={padding.left - 10}
                y={yPos + 4}
                fill="#6b7280"
                fontSize="11"
                fontWeight="600"
                textAnchor="end"
              >
                {value}
              </text>
            </g>
          );
        })}

        {/* Bars with modern styling */}
        {data.map((item, index) => {
          const barHeight = maxBarValue > 0 ? Math.max((item.value / maxBarValue) * chartHeight, 2) : 2;
          const xPosition = padding.left + (index * (barWidth + barSpacing));
          const color = colors[index % colors.length];
          const yPosition = barBottom - barHeight;
          
          // Calculate label spacing - more space for taller bars
          // Minimum 20px gap, but increase for taller bars
          const minGap = 20;
          const adaptiveGap = Math.max(minGap, barHeight * 0.1 + 15);
          const labelY = yPosition - adaptiveGap;
          const labelBackgroundY = labelY - 11;

          return (
            <g key={index} className="bar-group">
              {/* Shadow effect */}
              <rect
                x={xPosition + 2}
                y={yPosition + 2}
                width={barWidth}
                height={barHeight}
                fill="rgba(0,0,0,0.1)"
                rx="6"
                opacity="0.3"
              />
              
              {/* Main bar with gradient effect */}
              <defs>
                <linearGradient id={`gradient-${index}`} x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor={color} stopOpacity="1" />
                  <stop offset="100%" stopColor={color} stopOpacity="0.85" />
                </linearGradient>
              </defs>
              
              <rect
                x={xPosition}
                y={yPosition}
                width={barWidth}
                height={barHeight}
                fill={`url(#gradient-${index})`}
                rx="6"
                className="hover:opacity-90 transition-all duration-300"
                style={{ cursor: 'pointer' }}
              >
                <title>{item.label}: {item.value}</title>
              </rect>

              {/* Value label - always visible above the bar with background */}
              <g>
                {/* Background rectangle for text visibility */}
                <rect
                  x={xPosition + barWidth / 2 - (item.value.toString().length * 5 + 4)}
                  y={Math.max(labelBackgroundY, padding.top - 5)}
                  width={item.value.toString().length * 10 + 12}
                  height={22}
                  fill="rgba(255, 255, 255, 0.98)"
                  rx="5"
                  stroke={color}
                  strokeWidth="2"
                  style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' }}
                />
                <text
                  x={xPosition + barWidth / 2}
                  y={Math.max(labelY, padding.top + 8)}
                  fill={color}
                  fontSize="14"
                  fontWeight="bold"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{
                    pointerEvents: 'none',
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    letterSpacing: '0.5px'
                  }}
                >
                  {item.value}
                </text>
              </g>

              {/* X-axis label */}
              <text
                x={xPosition + barWidth / 2}
                y={barBottom + 25}
                fill="#374151"
                fontSize="12"
                fontWeight="600"
                textAnchor="middle"
                style={{ fontFamily: 'system-ui, sans-serif' }}
              >
                {item.label.length > 20 ? item.label.substring(0, 17) + '...' : item.label}
              </text>
            </g>
          );
        })}

        {/* Y-axis line */}
        <line
          x1={padding.left}
          y1={padding.top - 5}
          x2={padding.left}
          y2={barBottom + 5}
          stroke="#9ca3af"
          strokeWidth="2"
        />

        {/* X-axis line */}
        <line
          x1={padding.left - 5}
          y1={barBottom + 5}
          x2={chartWidth - padding.right + 5}
          y2={barBottom + 5}
          stroke="#9ca3af"
          strokeWidth="2"
        />
      </svg>
    </div>
  );
};

export default function DataAnalytics({ 
  results = [], 
  failedFiles = [],
  files = [],
  uploadStats = { total: 0, validated: 0, pending: 0 },
  districts = [],
  selectedMonth = '',
  selectedYear = '',
  selectedDistrict = '',
  selectedMandal = '',
  selectedVillage = ''
}) {
  const [filterYear, setFilterYear] = useState(selectedYear || '');
  const [filterMonth, setFilterMonth] = useState(selectedMonth || '');
  const [filterDistrict, setFilterDistrict] = useState('');
  const [filterMandal, setFilterMandal] = useState('');
  const [filterVillage, setFilterVillage] = useState('');
  const [showOverallState, setShowOverallState] = useState(true);

  // Data Capture Analytics state
  const [analyticsData, setAnalyticsData] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [analyticsError, setAnalyticsError] = useState('');
  const analyticsFileInputRef = useRef(null);
  const [chartView, setChartView] = useState('district'); // 'district' or 'year'
  const [chartType, setChartType] = useState('pie'); // 'bar' or 'pie'
  


  // Get filtered mandals and villages based on selected district
  const filteredMandals = useMemo(() => {
    if (!filterDistrict) return [];
    const district = districts.find(d => d.id === filterDistrict);
    return district?.mandals || [];
  }, [filterDistrict, districts]);

  const filteredVillages = useMemo(() => {
    if (!filterMandal) return [];
    const mandal = filteredMandals.find(m => m.id === filterMandal);
    return mandal?.villages || [];
  }, [filterMandal, filteredMandals]);


  const resetFilters = () => {
    setFilterYear('');
    setFilterMonth('');
    setFilterDistrict('');
    setFilterMandal('');
    setFilterVillage('');
  };

  // Fetch analytics data when filters change
  useEffect(() => {
    fetchAnalyticsData();
  }, [filterDistrict, filterMandal, filterVillage, filterMonth, filterYear]);

  // Refresh analytics data when component mounts
  useEffect(() => {
    fetchAnalyticsData();
  }, []); // Run once on mount

  const fetchAnalyticsData = async () => {
    try {
      setAnalyticsLoading(true);
      setAnalyticsError('');
      
      const params = new URLSearchParams();
      if (filterDistrict) {
        const dName = districts.find(d => d.id === filterDistrict)?.name;
        if (dName) params.append('district', dName);
      }
      if (filterMandal) {
        const mName = districts
          .find(d => d.id === filterDistrict)
          ?.mandals?.find(m => m.id === filterMandal)?.name;
        if (mName) params.append('mandal', mName);
      }
      if (filterVillage) {
      const district = districts.find(d => d.id === filterDistrict);
        const mandal = district?.mandals?.find(m => m.id === filterMandal);
        const vName = mandal?.villages?.find(v => v.id === filterVillage)?.name;
        if (vName) params.append('village', vName);
      }
      if (filterYear) params.append('year', filterYear);
      if (filterMonth) {
        const monthName = new Date(2000, parseInt(filterMonth) - 1).toLocaleString('default', { month: 'long' });
        params.append('month', monthName);
      }

      const res = await fetch(`${API_BASE}/api/analytics/data?${params.toString()}`);
      
      if (!res.ok) {
        let errorMessage = 'Unable to connect to backend.';
        try {
          const errorData = await res.json();
          errorMessage = errorData?.error || `Server returned ${res.status}: ${res.statusText}`;
        } catch (e) {
          errorMessage = `Server returned ${res.status}: ${res.statusText}. Please ensure the backend is running.`;
        }
        setAnalyticsData(null);
        setAnalyticsError(errorMessage);
        return;
      }

      const data = await res.json();

      if (!data.success) {
        console.error('Analytics API returned error:', data);
        setAnalyticsData(null);
        setAnalyticsError(data?.error || 'Unable to load analytics data. Please upload the Excel file.');
        return;
      }

      console.log('Analytics data received:', data.data);
      setAnalyticsData(data.data);
      setAnalyticsError('');
    } catch (error) {
      console.error("Failed to fetch analytics data:", error);
      setAnalyticsData(null);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        setAnalyticsError('Cannot connect to backend. Please ensure the Flask backend is running.');
      } else if (error.message) {
        setAnalyticsError(`Error: ${error.message}`);
      } else {
        setAnalyticsError('Failed to fetch analytics data. Please ensure the backend is running and an Excel file has been uploaded.');
      }
    } finally {
      setAnalyticsLoading(false);
    }
  };

  const handleAnalyticsFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setAnalyticsLoading(true);
      const res = await fetch(`${API_BASE}/api/analytics/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (res.ok) {
        const result = await res.json();
        if (result.success) {
          fetchAnalyticsData();
          alert("Analytics data uploaded successfully!");
        } else {
          alert("Upload failed: " + (result.error || 'Unknown error'));
        }
      } else {
        let errorMessage = `Upload failed with status ${res.status}`;
        try {
          const err = await res.json();
          errorMessage = err.error || errorMessage;
        } catch (e) {
          errorMessage = `Server returned ${res.status}: ${res.statusText}. Please ensure the backend is running.`;
        }
        alert(errorMessage);
      }
    } catch (error) {
      console.error("Upload error:", error);
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        alert("Cannot connect to backend. Please ensure the Flask backend is running.");
      } else {
        alert("Upload failed: " + (error.message || 'Unknown error'));
      }
    } finally {
      if (analyticsFileInputRef.current) {
        analyticsFileInputRef.current.value = "";
      }
      setAnalyticsLoading(false);
    }
  };

  const exportAnalytics = () => {
    if (!analyticsData) {
      alert('No analytics data available to export');
      return;
    }

    const data = {
      dataCaptureAnalytics: analyticsData,
      filters: {
        district: filterDistrict ? districts.find(d => d.id === filterDistrict)?.name : 'All',
        viewMode: showOverallState ? 'Overall State' : 'District Level'
      },
      generatedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `data_capture_analytics_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Render chart component
  const renderChart = (title, data, metricKey, colors) => {
    if (!data || Object.keys(data).length === 0) {
      return (
        <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
          <h5 className="text-lg font-bold text-gray-800 mb-4">{title}</h5>
          <div className="text-center py-8 text-gray-400">No data available</div>
        </div>
      );
    }

    // Prepare chart data
    const chartData = Object.entries(data)
      .map(([key, value]) => ({
        label: key.length > 20 ? key.substring(0, 17) + '...' : key,
        fullLabel: key,
        value: value[metricKey] || 0
      }))
      .filter(item => item.value > 0)
      .sort((a, b) => b.value - a.value);

    if (chartData.length === 0) {
      return (
        <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
          <h5 className="text-lg font-bold text-gray-800 mb-4">{title}</h5>
          <div className="text-center py-8 text-gray-400">No data available</div>
        </div>
      );
    }

    const maxValue = Math.max(...chartData.map(d => d.value), 1);

    return (
      <div className="bg-white rounded-xl p-6 border-2 border-gray-200 shadow-md transform transition-all duration-300 hover:shadow-xl hover:scale-[1.02] animate-fade-in-up">
        <h5 className="text-lg font-bold text-gray-800 mb-4">{title}</h5>
        {chartType === 'bar' ? (
          <BarChart
            data={chartData}
            maxValue={maxValue}
            height={300}
            colors={colors}
          />
        ) : (
          <PieChart
            data={chartData}
            size={280}
            colors={colors}
          />
        )}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-6 animate-fade-in">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg transform transition-all duration-300 hover:scale-110 hover:rotate-6">
            <BarChart3 size={28} className="text-white" />
          </div>
          <div>
            <h2 className="text-2xl lg:text-3xl font-bold text-gray-800 animate-fade-in">
              Data Analytics Dashboard
            </h2>
            <p className="text-sm text-gray-600 animate-fade-in" style={{ animationDelay: '0.2s' }}>Comprehensive insights and metrics</p>
          </div>
        </div>
       
      </div>

      {/* Filter Parameters Section - Moved to Top */}
      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-6 border-2 border-blue-300 mb-6 shadow-lg animate-fade-in">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-blue-100 rounded-lg">
          <Filter size={24} className="text-blue-600" />
          </div>
          <h3 className="text-xl font-bold text-blue-900">Filter Parameters</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
          {/* Year Filter */}
          <div className="animate-slide-in-up" style={{ animationDelay: '0.1s' }}>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              Year
            </label>
            <select
              value={filterYear}
              onChange={(e) => setFilterYear(e.target.value)}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white transition-all duration-300 hover:border-blue-400 hover:shadow-md"
            >
              <option value="">All Years</option>
              {Array.from({ length: 10 }, (_, i) => {
                const year = new Date().getFullYear() - i;
                return <option key={year} value={year}>{year}</option>;
              })}
            </select>
          </div>

          {/* Month Filter */}
          <div className="animate-slide-in-up" style={{ animationDelay: '0.2s' }}>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              Month
            </label>
            <select
              value={filterMonth}
              onChange={(e) => setFilterMonth(e.target.value)}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white transition-all duration-300 hover:border-blue-400 hover:shadow-md"
            >
              <option value="">All Months</option>
              <option value="01">January</option>
              <option value="02">February</option>
              <option value="03">March</option>
              <option value="04">April</option>
              <option value="05">May</option>
              <option value="06">June</option>
              <option value="07">July</option>
              <option value="08">August</option>
              <option value="09">September</option>
              <option value="10">October</option>
              <option value="11">November</option>
              <option value="12">December</option>
            </select>
          </div>

          {/* District Filter */}
          <div className="animate-slide-in-up" style={{ animationDelay: '0.3s' }}>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              District
            </label>
            <select
              value={filterDistrict}
              onChange={(e) => {
                setFilterDistrict(e.target.value);
                setFilterMandal('');
                setFilterVillage('');
              }}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white transition-all duration-300 hover:border-blue-400 hover:shadow-md"
            >
              <option value="">All Districts</option>
              {districts.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>

          {/* Mandal Filter */}
          <div className="animate-slide-in-up" style={{ animationDelay: '0.4s' }}>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              Mandal
            </label>
            <select
              value={filterMandal}
              onChange={(e) => {
                setFilterMandal(e.target.value);
                setFilterVillage('');
              }}
              disabled={!filterDistrict}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white disabled:bg-gray-100 disabled:cursor-not-allowed transition-all duration-300 hover:border-blue-400 hover:shadow-md disabled:hover:border-blue-300 disabled:hover:shadow-none"
            >
              <option value="">All Mandals</option>
              {filteredMandals.map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          </div>

          {/* Village Filter */}
          <div className="animate-slide-in-up" style={{ animationDelay: '0.5s' }}>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              Village
            </label>
            <select
              value={filterVillage}
              onChange={(e) => setFilterVillage(e.target.value)}
              disabled={!filterMandal}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white disabled:bg-gray-100 disabled:cursor-not-allowed transition-all duration-300 hover:border-blue-400 hover:shadow-md disabled:hover:border-blue-300 disabled:hover:shadow-none"
            >
              <option value="">All Villages</option>
              {filteredVillages.map((v) => (
                <option key={v.id} value={v.id}>{v.name}</option>
              ))}
            </select>
          </div>

          {/* Reset Button */}
          <div className="flex items-end animate-slide-in-up" style={{ animationDelay: '0.6s' }}>
            <button
              onClick={resetFilters}
              className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-xl font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg active:scale-95"
            >
              Reset Filters
            </button>
          </div>
        </div>
      </div>

      {/* Data Capture Analytics Section */}
      <div className="bg-white rounded-2xl border-2 border-gray-200 shadow-lg p-6 mb-6 animate-fade-in">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <TrendingUp size={24} className="text-indigo-600" />
            Data Capture Analytics
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => fetchAnalyticsData()}
              disabled={analyticsLoading}
              className="px-3 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 active:scale-95 flex items-center gap-2"
              title="Refresh analytics data"
            >
              <RotateCw size={18} className={analyticsLoading ? 'animate-spin' : ''} />
              Refresh
            </button>
          <button
              onClick={() => setShowOverallState(true)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                showOverallState 
                  ? 'bg-indigo-600 text-white shadow-lg' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Overall State
            </button>
            <button
              onClick={() => setShowOverallState(false)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                !showOverallState 
                  ? 'bg-indigo-600 text-white shadow-lg' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              District Level
          </button>
          </div>
        </div>

        {analyticsError && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-6">
            <div className="flex items-center gap-2">
              <AlertCircle size={20} />
              <span>{analyticsError}</span>
              </div>
          </div>
        )}

        {analyticsLoading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
            <p className="text-gray-600 font-medium">Loading analytics data...</p>
              </div>
        ) : !analyticsData ? (
          <div className="text-center py-12 bg-gray-50 rounded-xl border-2 border-dashed border-gray-300">
            <TrendingUp size={48} className="mx-auto text-gray-400 mb-4" />
            <p className="text-xl text-gray-600 font-semibold">No Analytics Data Found</p>
            <p className="text-gray-500 mt-2">Please ensure the analytics Excel file is available in the backend.</p>
            </div>
        ) : (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white shadow-lg transform transition-all duration-300 hover:scale-105 hover:shadow-2xl animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-blue-100 text-sm font-semibold">Total Imports</span>
                  <TrendingUp size={24} className="text-blue-200" />
              </div>
                <p className="text-3xl font-bold animate-count-up">{analyticsData.summary?.total_imports || 0}</p>
            </div>
              
              <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl p-6 text-white shadow-lg transform transition-all duration-300 hover:scale-105 hover:shadow-2xl animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-emerald-100 text-sm font-semibold">Validations Successful</span>
                  <CheckCircle size={24} className="text-emerald-200" />
              </div>
                <p className="text-3xl font-bold animate-count-up">{analyticsData.summary?.validation_successful || 0}</p>
            </div>
              
              <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-xl p-6 text-white shadow-lg transform transition-all duration-300 hover:scale-105 hover:shadow-2xl animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-red-100 text-sm font-semibold">Validations Failed</span>
                  <AlertTriangle size={24} className="text-red-200" />
              </div>
                <p className="text-3xl font-bold animate-count-up">{analyticsData.summary?.validation_failed || 0}</p>
            </div>
              
              <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white shadow-lg transform transition-all duration-300 hover:scale-105 hover:shadow-2xl animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-green-100 text-sm font-semibold">Synced to MKB</span>
                  <CheckCircle size={24} className="text-green-200" />
              </div>
                <p className="text-3xl font-bold animate-count-up">{analyticsData.summary?.synced_to_mkb || 0}</p>
            </div>
              
              <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white shadow-lg transform transition-all duration-300 hover:scale-105 hover:shadow-2xl animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-purple-100 text-sm font-semibold">Success Rate</span>
                  <BarChart3 size={24} className="text-purple-200" />
            </div>
                <p className="text-3xl font-bold animate-count-up">{analyticsData.summary?.success_rate || 0}%</p>
          </div>
      </div>

            {/* District Level or Overall State View */}
            {showOverallState ? (
              <div className="bg-gray-50 rounded-xl p-6 animate-fade-in">
                <h4 className="text-lg font-bold text-gray-800 mb-4">Overall State Summary</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm transform transition-all duration-300 hover:scale-105 hover:shadow-md">
                      <span className="font-semibold text-gray-700">Total Imports:</span>
                      <span className="text-2xl font-bold text-blue-600 animate-count-up">{analyticsData.summary?.total_imports || 0}</span>
          </div>
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm transform transition-all duration-300 hover:scale-105 hover:shadow-md">
                      <span className="font-semibold text-gray-700">Validations Successful:</span>
                      <span className="text-2xl font-bold text-emerald-600 animate-count-up">{analyticsData.summary?.validation_successful || 0}</span>
        </div>
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm transform transition-all duration-300 hover:scale-105 hover:shadow-md">
                      <span className="font-semibold text-gray-700">Validations Failed:</span>
                      <span className="text-2xl font-bold text-red-600 animate-count-up">{analyticsData.summary?.validation_failed || 0}</span>
          </div>
        </div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm transform transition-all duration-300 hover:scale-105 hover:shadow-md">
                      <span className="font-semibold text-gray-700">Synced to MKB:</span>
                      <span className="text-2xl font-bold text-green-600 animate-count-up">{analyticsData.summary?.synced_to_mkb || 0}</span>
          </div>
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm transform transition-all duration-300 hover:scale-105 hover:shadow-md">
                      <span className="font-semibold text-gray-700">Success Rate:</span>
                      <span className="text-2xl font-bold text-purple-600 animate-count-up">{analyticsData.summary?.success_rate || 0}%</span>
        </div>
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm transform transition-all duration-300 hover:scale-105 hover:shadow-md">
                      <span className="font-semibold text-gray-700">Validation Rate:</span>
                      <span className="text-2xl font-bold text-indigo-600 animate-count-up">
                        {analyticsData.summary?.total_imports > 0 
                          ? Math.round((analyticsData.summary?.validation_successful || 0) / analyticsData.summary.total_imports * 100)
                          : 0}%
                      </span>
          </div>
        </div>
      </div>
              </div>
            ) : (
              <div className="bg-gray-50 rounded-xl p-6 animate-fade-in">
                <h4 className="text-lg font-bold text-gray-800 mb-4">District Level Summary</h4>
                {Object.keys(analyticsData.district_summaries || {}).length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No district data available for selected filters
                      </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(analyticsData.district_summaries || {})
                      .sort(([, a], [, b]) => b.imports - a.imports)
                      .map(([district, data], index) => (
                      <div 
                        key={district} 
                        className="bg-white rounded-lg p-5 shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 animate-fade-in-up"
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <h5 className="font-bold text-gray-800 mb-3 text-lg border-b pb-2">{district}</h5>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Total Imports:</span>
                            <span className="font-semibold text-blue-600">{data.imports}</span>
                        </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Validations Failed:</span>
                            <span className="font-semibold text-red-600">{data.validation_failed}</span>
                    </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Synced to MKB:</span>
                            <span className="font-semibold text-green-600">{data.synced_to_mkb}</span>
                  </div>
                          <div className="flex justify-between items-center pt-2 border-t mt-2">
                            <span className="text-gray-700 font-semibold">Success Rate:</span>
                            <span className="font-bold text-lg text-purple-600">{data.success_rate}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Charts Section */}
            {analyticsData.chart_data && (
              <div className="space-y-6">
                {/* Chart View Toggle */}
                <div className="flex items-center justify-between bg-white rounded-lg p-4 border-2 border-gray-200 animate-fade-in shadow-md">
                  <h4 className="text-lg font-bold text-gray-800">Analytics Charts</h4>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setChartView('district')}
                      className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                        chartView === 'district'
                          ? 'bg-indigo-600 text-white shadow-lg'
                          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      }`}
                    >
                      By District
                    </button>
                    <button
                      onClick={() => setChartView('year')}
                      className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                        chartView === 'year'
                          ? 'bg-indigo-600 text-white shadow-lg'
                          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      }`}
                    >
                      By Year
                    </button>
                      </div>
                        </div>

                {/* Chart Type Toggle */}
                <div className="flex items-center justify-center gap-4 bg-white rounded-lg p-4 border-2 border-gray-200 animate-fade-in shadow-md">
                  <span className="font-semibold text-gray-700">Chart Type:</span>
                  <button
                    onClick={() => setChartType('bar')}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                      chartType === 'bar'
                        ? 'bg-blue-600 text-white shadow-lg'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    Bar Chart
                  </button>
                  <button
                    onClick={() => setChartType('pie')}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                      chartType === 'pie'
                        ? 'bg-blue-600 text-white shadow-lg'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    Pie Chart
                  </button>
                    </div>

                {/* Charts Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Total Forms Chart */}
                  {renderChart(
                    'Total Forms',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'total_forms',
                    ['#3b82f6', '#2563eb', '#1d4ed8', '#1e40af']
                  )}

                  {/* Validation Successful Chart */}
                  {renderChart(
                    'Validation Successful',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'validation_successful',
                    ['#10b981', '#059669', '#047857', '#065f46']
                  )}

                  {/* Validation Failed Chart */}
                  {renderChart(
                    'Validation Failed',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'validation_failed',
                    ['#ef4444', '#dc2626', '#b91c1c', '#991b1b']
                  )}

                  {/* Failed Incorrect Form Chart */}
                  {renderChart(
                    'Failed Incorrect Form',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'failed_incorrect_form',
                    ['#f97316', '#ea580c', '#c2410c', '#9a3412']
                  )}

                  {/* Failed Incorrect Values Chart */}
                  {renderChart(
                    'Failed Incorrect Values',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'failed_incorrect_values',
                    ['#f59e0b', '#d97706', '#b45309', '#92400e']
                  )}

                  {/* Failed Missing Fields Chart */}
                  {renderChart(
                    'Failed Missing Fields',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'failed_missing_fields',
                    ['#eab308', '#ca8a04', '#a16207', '#854d0e']
                  )}

                  {/* Failed Image Quality Chart */}
                  {renderChart(
                    'Failed Image Quality',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'failed_image_quality',
                    ['#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6']
                  )}

                  {/* Forms Synced to MBK Chart */}
                  {renderChart(
                    'Forms Synced to MBK',
                    analyticsData.chart_data[chartView === 'district' ? 'by_district' : 'by_year'],
                    'synced_to_mkb',
                    ['#14b8a6', '#0d9488', '#0f766e', '#115e59']
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

    </div>
  );
}

