import React, { useState, useEffect, useRef, useMemo } from 'react';
import { TrendingUp } from 'lucide-react';
import { BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import ApMap from './components/ApMap';
import apDistricts from './data/ap-districts';
import { API_BASE } from './utils/apiConfig';

const normalizeDistrictName = (name = '') => name.toLowerCase().replace(/[^a-z0-9]/g, '');

export default function FinancialAnalytics({
  districts = [],
  selectedMonth = '',
  selectedYear = '',
  selectedDistrict = '',
  selectedMandal = '',
  selectedVillage = ''
}) {
  const [financialData, setFinancialData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const financialFileInputRef = useRef(null);
  const [mapSelectedDistrictId, setMapSelectedDistrictId] = useState(null);
  const [mapSelectedDistrictName, setMapSelectedDistrictName] = useState('Andhra Pradesh');

  // Colors for charts
  const PIE_COLORS = ['#f59e0b', '#f43f5e', '#8b5cf6', '#10b981', '#1d4ed8', '#f97316', '#22c55e', '#a855f7'];
  const BAR_COLORS = {
    shg: '#f97316',      // Orange
    savings: '#6366f1',  // Indigo
    newLoan: '#34d399'   // Green
  };

  // Fetch financial data when tab is active or filters change
  useEffect(() => {
    fetchFinancialData();
  }, [selectedDistrict, selectedMandal, selectedVillage, selectedMonth, selectedYear]);

  const fetchFinancialData = async () => {
    try {
      setLoading(true);
      setError('');
      
      const params = new URLSearchParams();
      if (selectedDistrict) {
        const dName = districts.find(d => d.id === selectedDistrict)?.name;
        if (dName) params.append('district', dName);
      }
      if (selectedMandal) {
        const mName = districts
          .find(d => d.id === selectedDistrict)
          ?.mandals?.find(m => m.id === selectedMandal)?.name;
        if (mName) params.append('mandal', mName);
      }
      if (selectedVillage) {
        const district = districts.find(d => d.id === selectedDistrict);
        const mandal = district?.mandals?.find(m => m.id === selectedMandal);
        const vName = mandal?.villages?.find(v => v.id === selectedVillage)?.name;
        if (vName) params.append('village', vName);
      }
      if (selectedYear) params.append('year', selectedYear);
      if (selectedMonth) {
        const monthName = new Date(2000, parseInt(selectedMonth) - 1).toLocaleString('default', { month: 'long' });
        params.append('month', monthName);
      }

      const res = await fetch(`${API_BASE}/api/financial/data?${params.toString()}`);
      
      if (!res.ok) {
        let errorMessage = 'Unable to connect to backend.';
        try {
          const errorData = await res.json();
          errorMessage = errorData?.error || `Server returned ${res.status}: ${res.statusText}`;
        } catch (e) {
          errorMessage = `Server returned ${res.status}: ${res.statusText}. Please ensure the backend is running at ${API_BASE}`;
        }
        setFinancialData(null);
        setError(errorMessage);
        return;
      }

      const data = await res.json();

      if (!data.success) {
        setFinancialData(null);
        setError(data?.error || 'Unable to load financial data. Please upload the Excel file.');
        return;
      }

      setFinancialData(data.data);
      setError('');
    } catch (error) {
      console.error("Failed to fetch financial data:", error);
      setFinancialData(null);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        setError(`Cannot connect to backend. Please ensure the Flask backend is running at ${API_BASE}`);
      } else if (error.message) {
        setError(`Error: ${error.message}`);
      } else {
        setError('Failed to fetch financial data. Please ensure the backend is running and an Excel file has been uploaded.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFinancialFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/api/financial/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (res.ok) {
        const result = await res.json();
        if (result.success) {
          fetchFinancialData();
          alert("Financial data uploaded successfully!");
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
        alert(`Cannot connect to backend. Please ensure the Flask backend is running at ${API_BASE}`);
      } else {
        alert("Upload failed: " + (error.message || 'Unknown error'));
      }
    } finally {
      if (financialFileInputRef.current) {
        financialFileInputRef.current.value = "";
      }
      setLoading(false);
    }
  };

  // Handle map district selection
  const handleMapDistrictSelect = (districtId, districtName) => {
    setMapSelectedDistrictId(districtId);
    setMapSelectedDistrictName(districtName || 'Andhra Pradesh');
  };

  // Sync map selection with selected district from filters
  useEffect(() => {
    if (!selectedDistrict) {
      setMapSelectedDistrictId(null);
      setMapSelectedDistrictName('Andhra Pradesh');
      return;
    }
    const districtEntry = districts.find((district) => district.id === selectedDistrict);
    if (!districtEntry) return;
    const match = apDistricts.find(
      (apDistrict) => normalizeDistrictName(apDistrict.district) === normalizeDistrictName(districtEntry.name)
    );
    if (match) {
      setMapSelectedDistrictId(match.id);
      setMapSelectedDistrictName(match.district);
    }
  }, [selectedDistrict, districts]);

  // Prepare chart data from financial data
  const prepareChartData = () => {
    if (!financialData) return null;

    // Use district-wise data from backend and sort high-to-low for better readability
    const shgLoanData = (financialData.district_shg_loans || [])
      .slice()
      .sort((a, b) => (b.value || 0) - (a.value || 0));
    const loanTypeData = (financialData.loan_portfolio || [])
      .slice()
      .sort((a, b) => (b.value || 0) - (a.value || 0));
    const newLoanCountData = (financialData.district_new_loans || [])
      .slice()
      .sort((a, b) => (b.value || 0) - (a.value || 0));
    const savingsData = (financialData.district_savings || [])
      .slice()
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    return {
      shgLoanData,
      loanTypeData,
      newLoanCountData,
      savingsData
    };
  };

  const chartData = prepareChartData();

  // Prepare district summaries for map (similar to analytics)
  const districtSummaries = useMemo(() => {
    if (!financialData?.district_summaries) return {};
    return financialData.district_summaries;
  }, [financialData]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6 mb-6">
          <div>
            <h2 className="text-2xl lg:text-3xl font-bold text-gray-800 flex items-center gap-2">
              <TrendingUp size={32} className="text-indigo-600" />
              SHG Financial Analytics
            </h2>
            <p className="text-gray-500 mt-2">
              {selectedDistrict || "All Districts"} 
              {selectedMandal && ` > ${selectedMandal}`} 
              {selectedVillage && ` > ${selectedVillage}`}
              {(selectedMonth || selectedYear) && ` | ${selectedMonth} ${selectedYear}`}
            </p>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-6">
            {error}
          </div>
        )}

        {loading ? (
          <div className="text-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
            <p className="text-gray-600 font-medium">Loading financial data...</p>
          </div>
        ) : !financialData ? null : (
          <div className="space-y-6">
            {/* Interactive District Map */}
            <div className="bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-gray-800">Interactive District Insights Map</h3>
                  <p className="text-gray-500">
                    Click a district to view mandals, villages, and financial details. Use this alongside the analytics below to
                    understand regional coverage.
                  </p>
                </div>
              </div>
              <ApMap
                selectedDistrictId={mapSelectedDistrictId}
                onDistrictSelect={handleMapDistrictSelect}
                districtSummaries={districtSummaries}
                isAnalyticsLoading={loading}
              />
            </div>

            {/* Charts Section */}
            <div className="space-y-8">
            {/* Total - SHG Loan Balance by District */}
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-lg">
              <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">
                Total - SHG Loan Balance by District
              </h3>
              <div className="h-96 w-full min-h-[360px]">
                {chartData?.shgLoanData && chartData.shgLoanData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%" minHeight={320}>
                    <BarChart
                      data={chartData.shgLoanData}
                      margin={{ top: 20, right: 20, left: 10, bottom: 40 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0} height={70} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill={BAR_COLORS.shg} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400 font-semibold">
                    No SHG loan balance data
                  </div>
                )}
              </div>
            </div>

            {/* Loan Type Distribution */}
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-lg">
              <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Loan Type Distribution</h3>
              <div className="h-96 w-full min-h-[360px]">
                {financialData.loan_type_distribution && financialData.loan_type_distribution.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%" minHeight={320}>
                    <PieChart>
                      <Pie
                        data={financialData.loan_type_distribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={110}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {financialData.loan_type_distribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400 font-semibold">
                    No loan type data
                  </div>
                )}
              </div>
            </div>

            {/* Total Savings Balance by District */}
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-lg">
              <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">
                Total Savings Balance by District
              </h3>
              <div className="h-96 w-full min-h-[360px]">
                {chartData?.savingsData && chartData.savingsData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%" minHeight={320}>
                    <BarChart
                      data={chartData.savingsData}
                      margin={{ top: 20, right: 20, left: 10, bottom: 40 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0} height={70} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill={BAR_COLORS.savings} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400 font-semibold">
                    No savings data
                  </div>
                )}
              </div>
            </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

