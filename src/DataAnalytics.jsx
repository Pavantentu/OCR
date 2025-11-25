import React, { useState, useMemo } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, CheckCircle, X, Filter, Download, MapPin } from 'lucide-react';

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
                  y={Math.max(yPosition - 30, padding.top - 5)}
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
                  y={Math.max(yPosition - 17, padding.top + 8)}
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

  // Calculate statistics
  const analytics = useMemo(() => {
    // Filter results and failed files based on selected filters
    let filteredResults = [...results];
    let filteredFailed = [...failedFiles];

    // Filter by year
    if (filterYear) {
      filteredResults = filteredResults.filter(r => String(r.year) === String(filterYear));
      filteredFailed = filteredFailed.filter(f => String(f.year) === String(filterYear));
    }

    // Filter by month
    if (filterMonth) {
      const monthName = new Date(2000, parseInt(filterMonth) - 1).toLocaleString('default', { month: 'long' });
      filteredResults = filteredResults.filter(r => r.month === monthName);
      filteredFailed = filteredFailed.filter(f => f.month === monthName);
    }

    // Filter by district
    if (filterDistrict) {
      const districtName = districts.find(d => d.id === filterDistrict)?.name;
      if (districtName) {
        filteredResults = filteredResults.filter(r => r.district === districtName);
        filteredFailed = filteredFailed.filter(f => f.district === districtName);
      }
    }

    // Filter by mandal
    if (filterMandal) {
      let mandalName = null;
      if (filterDistrict) {
        const district = districts.find(d => d.id === filterDistrict);
        if (district) {
          const mandal = district.mandals?.find(m => m.id === filterMandal);
          mandalName = mandal?.name;
        }
      }
      if (mandalName) {
        filteredResults = filteredResults.filter(r => r.mandal === mandalName);
        filteredFailed = filteredFailed.filter(f => f.mandal === mandalName);
      }
    }

    // Filter by village
    if (filterVillage) {
      let villageName = null;
      if (filterMandal && filterDistrict) {
        const district = districts.find(d => d.id === filterDistrict);
        if (district) {
          const mandal = district.mandals?.find(m => m.id === filterMandal);
          if (mandal) {
            const village = mandal.villages?.find(v => v.id === filterVillage);
            villageName = village?.name;
          }
        }
      }
      if (villageName) {
        filteredResults = filteredResults.filter(r => r.village === villageName);
        filteredFailed = filteredFailed.filter(f => f.village === villageName);
      }
    }

    // Get unique file groups from processed results
    const fileGroups = new Set();
    filteredResults.forEach(r => {
      const key = r.fileGroupId || `${r.filename}-${r.month}-${r.year}`;
      fileGroups.add(key);
    });

    // Count files in upload queue (pending processing)
    const pendingValidationFiles = files.filter(f => !f.validated).length;
    const validatedPendingFiles = files.filter(f => f.validated).length;
    
    // Calculate totals including current upload queue
    const processedFormsCount = fileGroups.size;
    const failedFormsCount = filteredFailed.length;
    const totalFormsUploaded = processedFormsCount + failedFormsCount + files.length;
    const validationsSuccessful = processedFormsCount;
    const formFailedToCapture = failedFormsCount;
    const pendingValidation = pendingValidationFiles;
    const validatedPendingProcessing = validatedPendingFiles;

    // Categorize failed files with better error detection
    const failedSummary = {
      incorrectForm: filteredFailed.filter(f => {
        const error = f.error?.toLowerCase() || '';
        return error.includes('incorrect') || 
               error.includes('invalid') ||
               error.includes('wrong') ||
               error.includes('no tables found') ||
               error.includes('table extraction failed');
      }).length,
      incorrectValues: filteredFailed.filter(f => {
        const error = f.error?.toLowerCase() || '';
        return error.includes('value') ||
               error.includes('data') ||
               error.includes('parse') ||
               error.includes('format');
      }).length,
      missingFields: filteredFailed.filter(f => {
        const error = f.error?.toLowerCase() || '';
        return error.includes('missing') ||
               error.includes('field') ||
               error.includes('required') ||
               error.includes('empty');
      }).length,
      connectionErrors: filteredFailed.filter(f => {
        const error = f.error?.toLowerCase() || '';
        return error.includes('connection') ||
               error.includes('server') ||
               error.includes('backend');
      }).length
    };

    // Forms synced to MBK (successful conversions)
    const totalFormsSyncedToMBK = validationsSuccessful;
    // Pending forms = failed + pending validation + validated but not processed
    const pendingForms = formFailedToCapture + pendingValidation + validatedPendingProcessing;
    
    // Calculate confidence (success rate) for processed forms
    const processedTotal = processedFormsCount + failedFormsCount;
    const confidence = processedTotal > 0 
      ? ((validationsSuccessful / processedTotal) * 100).toFixed(2)
      : 0;

    return {
      totalFormsUploaded,
      validationsSuccessful,
      formFailedToCapture,
      pendingValidation,
      validatedPendingProcessing,
      failedSummary,
      totalFormsSyncedToMBK,
      pendingForms,
      filteredResults,
      filteredFailed,
      processedFormsCount,
      failedFormsCount,
      totalWithPending: totalFormsUploaded,
      confidence: parseFloat(confidence)
    };
  }, [results, failedFiles, files, filterYear, filterMonth, filterDistrict, filterMandal, filterVillage, districts]);

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

  // Overall state summary (all data without filters)
  const overallStateSummary = useMemo(() => {
    const allFileGroups = new Set();
    results.forEach(r => {
      const key = r.fileGroupId || `${r.filename}-${r.month}-${r.year}`;
      allFileGroups.add(key);
    });

    const processedTotal = allFileGroups.size + failedFiles.length;
    const successful = allFileGroups.size;
    const failed = failedFiles.length;
    const totalWithPending = processedTotal + files.length;

    // Get unique districts, mandals, villages
    const uniqueDistricts = new Set(results.map(r => r.district).filter(Boolean));
    const uniqueMandals = new Set(results.map(r => r.mandal).filter(Boolean));
    const uniqueVillages = new Set(results.map(r => r.village).filter(Boolean));

    return {
      totalForms: totalWithPending,
      processedForms: processedTotal,
      successfulForms: successful,
      failedForms: failed,
      pendingForms: files.length,
      districtsCount: uniqueDistricts.size,
      mandalsCount: uniqueMandals.size,
      villagesCount: uniqueVillages.size
    };
  }, [results, failedFiles, files]);

  const resetFilters = () => {
    setFilterYear('');
    setFilterMonth('');
    setFilterDistrict('');
    setFilterMandal('');
    setFilterVillage('');
  };

  const exportAnalytics = () => {
    let mandalName = 'All';
    let villageName = 'All';
    
    if (filterMandal && filterDistrict) {
      const district = districts.find(d => d.id === filterDistrict);
      if (district) {
        const mandal = district.mandals?.find(m => m.id === filterMandal);
        mandalName = mandal?.name || 'All';
        
        if (filterVillage && mandal) {
          const village = mandal.villages?.find(v => v.id === filterVillage);
          villageName = village?.name || 'All';
        }
      }
    }

    const data = {
      filters: {
        year: filterYear || 'All',
        month: filterMonth ? new Date(2000, parseInt(filterMonth) - 1).toLocaleString('default', { month: 'long' }) : 'All',
        district: filterDistrict ? districts.find(d => d.id === filterDistrict)?.name : 'All',
        mandal: mandalName,
        village: villageName
      },
      metrics: analytics,
      overallState: overallStateSummary,
      generatedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analytics_${filterYear || 'all'}_${filterMonth || 'all'}_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
            <BarChart3 size={28} className="text-white" />
          </div>
          <div>
            <h2 className="text-2xl lg:text-3xl font-bold text-gray-800">
              Data Analytics Dashboard
            </h2>
            <p className="text-sm text-gray-600">Comprehensive insights and metrics</p>
          </div>
        </div>
        <button
          onClick={exportAnalytics}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold flex items-center gap-2"
        >
          <Download size={20} />
          Export
        </button>
      </div>

      {/* Parameters Section */}
      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-6 border-2 border-blue-300 mb-6 shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <Filter size={24} className="text-blue-600" />
          <h3 className="text-xl font-bold text-blue-900">Filter Parameters</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
          {/* Year Filter */}
          <div>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              Year
            </label>
            <select
              value={filterYear}
              onChange={(e) => setFilterYear(e.target.value)}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              <option value="">All Years</option>
              {Array.from({ length: 10 }, (_, i) => {
                const year = new Date().getFullYear() - i;
                return <option key={year} value={year}>{year}</option>;
              })}
            </select>
          </div>

          {/* Month Filter */}
          <div>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              Month
            </label>
            <select
              value={filterMonth}
              onChange={(e) => setFilterMonth(e.target.value)}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
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
          <div>
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
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              <option value="">All Districts</option>
              {districts.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>

          {/* Mandal Filter */}
          <div>
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
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white disabled:bg-gray-100 disabled:cursor-not-allowed"
            >
              <option value="">All Mandals</option>
              {filteredMandals.map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          </div>

          {/* Village Filter */}
          <div>
            <label className="block text-sm font-bold text-blue-900 mb-2">
              Village
            </label>
            <select
              value={filterVillage}
              onChange={(e) => setFilterVillage(e.target.value)}
              disabled={!filterMandal}
              className="w-full px-4 py-2 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white disabled:bg-gray-100 disabled:cursor-not-allowed"
            >
              <option value="">All Villages</option>
              {filteredVillages.map((v) => (
                <option key={v.id} value={v.id}>{v.name}</option>
              ))}
            </select>
          </div>

          {/* Reset Button */}
          <div className="flex items-end">
            <button
              onClick={resetFilters}
              className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-xl font-semibold"
            >
              Reset Filters
            </button>
          </div>
        </div>
      </div>

      {/* Overall State Summary Toggle */}
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl p-6 border-2 border-purple-300 mb-6 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <TrendingUp size={24} className="text-purple-600" />
            <h3 className="text-xl font-bold text-purple-900">Overall State Summary</h3>
          </div>
          <button
            onClick={() => setShowOverallState(!showOverallState)}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold"
          >
            {showOverallState ? 'Hide' : 'Show'}
          </button>
        </div>

        {showOverallState && (
          <div className="mt-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
              <div className="bg-white rounded-xl p-6 border-2 border-purple-200">
                <h4 className="text-lg font-bold text-gray-800 mb-4 text-center">Overall Status Distribution</h4>
                <PieChart
                  data={[
                    { label: 'Successful', value: overallStateSummary.successfulForms },
                    { label: 'Failed', value: overallStateSummary.failedForms },
                    { label: 'Pending', value: overallStateSummary.pendingForms || 0 }
                  ]}
                  size={280}
                  colors={['#10b981', '#ef4444', '#f59e0b']}
                />
              </div>
              <div className="bg-white rounded-xl p-6 border-2 border-purple-200">
                <h4 className="text-lg font-bold text-gray-800 mb-4 text-center">Overall Status Comparison</h4>
                <BarChart
                  data={[
                    { label: 'Total Forms', value: overallStateSummary.totalForms },
                    { label: 'Successful', value: overallStateSummary.successfulForms },
                    { label: 'Failed', value: overallStateSummary.failedForms },
                    { label: 'Pending', value: overallStateSummary.pendingForms || 0 }
                  ]}
                  maxValue={Math.max(
                    overallStateSummary.totalForms,
                    overallStateSummary.successfulForms,
                    overallStateSummary.failedForms,
                    overallStateSummary.pendingForms || 0,
                    1
                  )}
                  height={280}
                  colors={['#a855f7', '#10b981', '#ef4444', '#f59e0b']}
                />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl p-4 border-2 border-purple-200 text-center">
              <div className="text-3xl font-bold text-purple-800 mb-1">
                {overallStateSummary.totalForms}
              </div>
              <div className="text-sm font-semibold text-purple-600">Total Forms</div>
            </div>
            <div className="bg-white rounded-xl p-4 border-2 border-green-200 text-center">
              <div className="text-3xl font-bold text-green-800 mb-1">
                {overallStateSummary.successfulForms}
              </div>
              <div className="text-sm font-semibold text-green-600">Successful</div>
            </div>
            <div className="bg-white rounded-xl p-4 border-2 border-orange-200 text-center">
              <div className="text-3xl font-bold text-orange-800 mb-1">
                {overallStateSummary.pendingForms || 0}
              </div>
              <div className="text-sm font-semibold text-orange-600">Pending</div>
            </div>
            <div className="bg-white rounded-xl p-4 border-2 border-blue-200 text-center">
              <div className="text-3xl font-bold text-blue-800 mb-1">
                {overallStateSummary.districtsCount}
              </div>
              <div className="text-sm font-semibold text-blue-600">Districts</div>
            </div>
            </div>
          </div>
        )}
      </div>

      {/* Key Metrics Section */}
      <div className="space-y-6">
        {/* Metric 1: Total Forms Uploaded, Validations Successful, Form failed to capture */}
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-6 border-2 border-green-300 shadow-lg">
          <h3 className="text-xl font-bold text-green-900 mb-6 flex items-center gap-2">
            <CheckCircle size={24} className="text-green-600" />
            Total Forms Uploaded, Validations Successful, Form failed to capture
          </h3>
          <div className="bg-white rounded-xl p-6 border-2 border-green-200">
            <BarChart
              data={[
                { label: 'Total Uploaded', value: analytics.totalFormsUploaded },
                { label: 'Validations Successful', value: analytics.validationsSuccessful },
                { label: 'Failed to Capture', value: analytics.formFailedToCapture }
              ]}
              maxValue={Math.max(
                analytics.totalFormsUploaded, 
                analytics.validationsSuccessful, 
                analytics.formFailedToCapture,
                1
              )}
              height={300}
              colors={['#3b82f6', '#10b981', '#ef4444']}
            />
          </div>
        </div>

        {/* Metric 2: Form failed capture summary - Incorrect Form, Incorrect Values, Missing Fields */}
        <div className="bg-gradient-to-br from-red-50 to-rose-50 rounded-2xl p-6 border-2 border-red-300 shadow-lg">
          <h3 className="text-xl font-bold text-red-900 mb-6 flex items-center gap-2">
            <AlertTriangle size={24} className="text-red-600" />
            Form failed capture summary - Incorrect Form, Incorrect Values, Missing Fields
          </h3>
          <div className="bg-white rounded-xl p-6 border-2 border-red-200">
            <BarChart
              data={[
                { label: 'Incorrect Form', value: analytics.failedSummary.incorrectForm },
                { label: 'Incorrect Values', value: analytics.failedSummary.incorrectValues },
                { label: 'Missing Fields', value: analytics.failedSummary.missingFields }
              ]}
              maxValue={Math.max(
                analytics.failedSummary.incorrectForm,
                analytics.failedSummary.incorrectValues,
                analytics.failedSummary.missingFields,
                1
              )}
              height={300}
              colors={['#f97316', '#eab308', '#f59e0b']}
            />
          </div>
        </div>

        {/* Metric 3: Confidence of Digital Conversion */}
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl p-6 border-2 border-indigo-300 shadow-lg">
          <h3 className="text-xl font-bold text-indigo-900 mb-6 flex items-center gap-2">
            <TrendingUp size={24} className="text-indigo-600" />
            Confidence of Digital Conversion
          </h3>
          <div className="bg-white rounded-xl p-6 border-2 border-indigo-200 flex justify-center">
            <PieChart
              data={[
                { label: 'Successful', value: analytics.validationsSuccessful },
                { label: 'Failed', value: analytics.formFailedToCapture }
              ]}
              size={300}
              colors={['#10b981', '#ef4444']}
            />
          </div>
        </div>

        {/* Metric 4: Total Forms Synced to MBK, Pending Forms */}
        <div className="bg-gradient-to-br from-cyan-50 to-teal-50 rounded-2xl p-6 border-2 border-cyan-300 shadow-lg">
          <h3 className="text-xl font-bold text-cyan-900 mb-6 flex items-center gap-2">
            <CheckCircle size={24} className="text-cyan-600" />
            Total Forms Synced to MBK, Pending Forms
          </h3>
          <div className="bg-white rounded-xl p-6 border-2 border-cyan-200">
            <BarChart
              data={[
                { label: 'Synced to MBK', value: analytics.totalFormsSyncedToMBK },
                { label: 'Pending Forms', value: analytics.pendingForms }
              ]}
              maxValue={Math.max(analytics.totalFormsSyncedToMBK, analytics.pendingForms, 1)}
              height={300}
              colors={['#10b981', '#f97316']}
            />
          </div>
        </div>
      </div>

      {/* District, Mandal, and Village Breakdown Section */}
      {(() => {
        // Get selected district, mandal, village names
        const selectedDistrictName = filterDistrict ? districts.find(d => d.id === filterDistrict)?.name : null;
        const selectedMandalName = filterMandal && filterDistrict ? 
          districts.find(d => d.id === filterDistrict)?.mandals?.find(m => m.id === filterMandal)?.name : null;
        const selectedVillageName = filterVillage && filterMandal && filterDistrict ?
          districts.find(d => d.id === filterDistrict)?.mandals?.find(m => m.id === filterMandal)?.villages?.find(v => v.id === filterVillage)?.name : null;
        const monthName = filterMonth ? new Date(2000, parseInt(filterMonth) - 1).toLocaleString('default', { month: 'long' }) : null;

        // Filter data based on selections
        let displayResults = [...analytics.filteredResults];
        let displayFailed = [...analytics.filteredFailed];

        // Additional filtering for breakdowns - only show selected district/mandal/village
        if (selectedDistrictName) {
          displayResults = displayResults.filter(r => r.district === selectedDistrictName);
          displayFailed = displayFailed.filter(f => f.district === selectedDistrictName);
        }
        if (selectedMandalName) {
          displayResults = displayResults.filter(r => r.mandal === selectedMandalName);
          displayFailed = displayFailed.filter(f => f.mandal === selectedMandalName);
        }
        if (selectedVillageName) {
          displayResults = displayResults.filter(r => r.village === selectedVillageName);
          displayFailed = displayFailed.filter(f => f.village === selectedVillageName);
        }

        // Calculate stats for selected location
        const districtStats = {};
        displayResults.forEach(r => {
          if (r.district) {
            if (!districtStats[r.district]) {
              districtStats[r.district] = { successful: 0, failed: 0 };
            }
            districtStats[r.district].successful++;
          }
        });
        displayFailed.forEach(f => {
          if (f.district) {
            if (!districtStats[f.district]) {
              districtStats[f.district] = { successful: 0, failed: 0 };
            }
            districtStats[f.district].failed++;
          }
        });

        const mandalStats = {};
        displayResults.forEach(r => {
          if (r.mandal) {
            const key = `${r.district || 'Unknown'}-${r.mandal}`;
            if (!mandalStats[key]) {
              mandalStats[key] = { mandal: r.mandal, district: r.district || 'Unknown', successful: 0, failed: 0 };
            }
            mandalStats[key].successful++;
          }
        });
        displayFailed.forEach(f => {
          if (f.mandal) {
            const key = `${f.district || 'Unknown'}-${f.mandal}`;
            if (!mandalStats[key]) {
              mandalStats[key] = { mandal: f.mandal, district: f.district || 'Unknown', successful: 0, failed: 0 };
            }
            mandalStats[key].failed++;
          }
        });

        const villageStats = {};
        displayResults.forEach(r => {
          if (r.village) {
            const key = `${r.district || 'Unknown'}-${r.mandal || 'Unknown'}-${r.village}`;
            if (!villageStats[key]) {
              villageStats[key] = { village: r.village, mandal: r.mandal || 'Unknown', district: r.district || 'Unknown', successful: 0, failed: 0 };
            }
            villageStats[key].successful++;
          }
        });
        displayFailed.forEach(f => {
          if (f.village) {
            const key = `${f.district || 'Unknown'}-${f.mandal || 'Unknown'}-${f.village}`;
            if (!villageStats[key]) {
              villageStats[key] = { village: f.village, mandal: f.mandal || 'Unknown', district: f.district || 'Unknown', successful: 0, failed: 0 };
            }
            villageStats[key].failed++;
          }
        });

        const districtData = Object.entries(districtStats).map(([name, stats]) => ({
          label: name,
          successful: stats.successful,
          failed: stats.failed,
          total: stats.successful + stats.failed
        })).sort((a, b) => b.total - a.total);

        const mandalData = Object.values(mandalStats).map(stats => ({
          label: stats.mandal,
          district: stats.district,
          successful: stats.successful,
          failed: stats.failed,
          total: stats.successful + stats.failed
        })).sort((a, b) => b.total - a.total);

        const villageData = Object.values(villageStats).map(stats => ({
          label: stats.village,
          mandal: stats.mandal,
          district: stats.district,
          successful: stats.successful,
          failed: stats.failed,
          total: stats.successful + stats.failed
        })).sort((a, b) => b.total - a.total);

        const filterInfo = [];
        if (filterYear) filterInfo.push(`Year: ${filterYear}`);
        if (monthName) filterInfo.push(`Month: ${monthName}`);
        if (selectedDistrictName) filterInfo.push(`District: ${selectedDistrictName}`);
        if (selectedMandalName) filterInfo.push(`Mandal: ${selectedMandalName}`);
        if (selectedVillageName) filterInfo.push(`Village: ${selectedVillageName}`);

        return (
          <div className="mt-6 space-y-6">
            {filterInfo.length > 0 && (
              <div className="bg-blue-100 rounded-xl p-4 border-2 border-blue-300">
                <p className="text-sm font-semibold text-blue-900">
                  Showing data for: {filterInfo.join(' | ')}
                </p>
              </div>
            )}

            {/* District Breakdown - only show if district is selected or show all if none selected */}
            {(!selectedDistrictName || districtData.length > 0) && (
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-6 border-2 border-blue-300 shadow-lg">
                <h3 className="text-xl font-bold text-blue-900 mb-6 flex items-center gap-2">
                  <MapPin size={24} className="text-blue-600" />
                  Data by District {selectedDistrictName ? `(${selectedDistrictName})` : ''} {monthName ? `- ${monthName} ${filterYear || ''}` : ''}
                </h3>
                {districtData.length > 0 ? (
                  <div className="bg-white rounded-xl p-6 border-2 border-blue-200">
                    {districtData.length > 1 ? (
                      <BarChart
                        data={districtData.map(d => ({ label: d.label.length > 20 ? d.label.substring(0, 17) + '...' : d.label, value: d.total }))}
                        maxValue={Math.max(...districtData.map(d => d.total), 1)}
                        height={300}
                        colors={districtData.map((_, i) => ['#3b82f6', '#6366f1', '#8b5cf6', '#a855f7'][i % 4])}
                      />
                    ) : (
                      <div className="text-center py-8">
                        <div className="text-4xl font-bold text-blue-800 mb-2">{districtData[0].total}</div>
                        <div className="text-lg text-gray-600">Total Forms</div>
                      </div>
                    )}
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {districtData.map((d, idx) => (
                        <div key={idx} className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                          <div className="font-bold text-blue-900 text-sm mb-2">{d.label}</div>
                          <div className="text-xs text-gray-600">Successful: <span className="font-semibold text-green-700">{d.successful}</span></div>
                          <div className="text-xs text-gray-600">Failed: <span className="font-semibold text-red-700">{d.failed}</span></div>
                          <div className="text-xs text-gray-600">Total: <span className="font-semibold text-blue-700">{d.total}</span></div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="bg-white rounded-xl p-6 border-2 border-blue-200 text-center text-gray-500">
                    No district data available for selected filters
                  </div>
                )}
              </div>
            )}

            {/* Mandal Breakdown - only show if mandal is selected or district is selected */}
            {(!selectedMandalName || mandalData.length > 0) && (
              <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl p-6 border-2 border-purple-300 shadow-lg">
                <h3 className="text-xl font-bold text-purple-900 mb-6 flex items-center gap-2">
                  <MapPin size={24} className="text-purple-600" />
                  Data by Mandal {selectedMandalName ? `(${selectedMandalName})` : selectedDistrictName ? `(${selectedDistrictName})` : ''} {monthName ? `- ${monthName} ${filterYear || ''}` : ''}
                </h3>
                {mandalData.length > 0 ? (
                  <div className="bg-white rounded-xl p-6 border-2 border-purple-200">
                    {mandalData.length > 1 ? (
                      <BarChart
                        data={mandalData.map(d => ({ label: d.label.length > 20 ? d.label.substring(0, 17) + '...' : d.label, value: d.total }))}
                        maxValue={Math.max(...mandalData.map(d => d.total), 1)}
                        height={300}
                        colors={mandalData.map((_, i) => ['#a855f7', '#9333ea', '#7e22ce', '#6b21a8'][i % 4])}
                      />
                    ) : (
                      <div className="text-center py-8">
                        <div className="text-4xl font-bold text-purple-800 mb-2">{mandalData[0].total}</div>
                        <div className="text-lg text-gray-600">Total Forms</div>
                      </div>
                    )}
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {mandalData.map((d, idx) => (
                        <div key={idx} className="bg-purple-50 rounded-lg p-3 border border-purple-200">
                          <div className="font-bold text-purple-900 text-sm mb-1">{d.label}</div>
                          {!selectedDistrictName && <div className="text-xs text-gray-500 mb-2">({d.district})</div>}
                          <div className="text-xs text-gray-600">Successful: <span className="font-semibold text-green-700">{d.successful}</span></div>
                          <div className="text-xs text-gray-600">Failed: <span className="font-semibold text-red-700">{d.failed}</span></div>
                          <div className="text-xs text-gray-600">Total: <span className="font-semibold text-purple-700">{d.total}</span></div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="bg-white rounded-xl p-6 border-2 border-purple-200 text-center text-gray-500">
                    No mandal data available for selected filters
                  </div>
                )}
              </div>
            )}

            {/* Village Breakdown - only show if village is selected or mandal/district is selected */}
            {(!selectedVillageName || villageData.length > 0) && (
              <div className="bg-gradient-to-br from-teal-50 to-cyan-50 rounded-2xl p-6 border-2 border-teal-300 shadow-lg">
                <h3 className="text-xl font-bold text-teal-900 mb-6 flex items-center gap-2">
                  <MapPin size={24} className="text-teal-600" />
                  Data by Village {selectedVillageName ? `(${selectedVillageName})` : selectedMandalName ? `(${selectedMandalName})` : ''} {monthName ? `- ${monthName} ${filterYear || ''}` : ''}
                </h3>
                {villageData.length > 0 ? (
                  <div className="bg-white rounded-xl p-6 border-2 border-teal-200">
                    {villageData.length > 1 ? (
                      <BarChart
                        data={villageData.map(d => ({ label: d.label.length > 20 ? d.label.substring(0, 17) + '...' : d.label, value: d.total }))}
                        maxValue={Math.max(...villageData.map(d => d.total), 1)}
                        height={300}
                        colors={villageData.map((_, i) => ['#14b8a6', '#0d9488', '#0f766e', '#115e59'][i % 4])}
                      />
                    ) : (
                      <div className="text-center py-8">
                        <div className="text-4xl font-bold text-teal-800 mb-2">{villageData[0].total}</div>
                        <div className="text-lg text-gray-600">Total Forms</div>
                      </div>
                    )}
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {villageData.map((d, idx) => (
                        <div key={idx} className="bg-teal-50 rounded-lg p-3 border border-teal-200">
                          <div className="font-bold text-teal-900 text-sm mb-1">{d.label}</div>
                          {!selectedMandalName && <div className="text-xs text-gray-500 mb-2">{d.mandal} → {d.district}</div>}
                          <div className="text-xs text-gray-600">Successful: <span className="font-semibold text-green-700">{d.successful}</span></div>
                          <div className="text-xs text-gray-600">Failed: <span className="font-semibold text-red-700">{d.failed}</span></div>
                          <div className="text-xs text-gray-600">Total: <span className="font-semibold text-teal-700">{d.total}</span></div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="bg-white rounded-xl p-6 border-2 border-teal-200 text-center text-gray-500">
                    No village data available for selected filters
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}

