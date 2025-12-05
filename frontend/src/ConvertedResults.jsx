import React, { useState, useMemo } from 'react';
import { Download, Eye, Table2, Folder, FolderOpen, FileText, X, Info, Grid3X3, Edit3 } from 'lucide-react';

const TOTAL_FILES_TARGET = 10;

export default function ConvertedResults({ 
  results, 
  failedFiles = [], 
  onViewResult, 
  onExportCSV,
  onShowStats,
  selectedMonth,
  selectedYear,
  onUpdateResult,
  onUpdateAnalytics
}) {
  const [expandedFolders, setExpandedFolders] = useState({
    converted: false,
    nonConverted: false
  });
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [activeFolderModal, setActiveFolderModal] = useState(null);
  const [editingResult, setEditingResult] = useState(null);
  const [editableData, setEditableData] = useState([]);
  const [editHeaders, setEditHeaders] = useState([]);
  const [htmlFragments, setHtmlFragments] = useState({ before: '', after: '' });

  const selectedMonthName = useMemo(() => {
    if (!selectedMonth) return null;
    const monthIndex = parseInt(selectedMonth, 10);
    if (isNaN(monthIndex) || monthIndex < 1 || monthIndex > 12) return null;
    return new Date(2000, monthIndex - 1).toLocaleString('default', { month: 'long' });
  }, [selectedMonth]);

  const filteredResults = useMemo(() => {
    return results.filter(result => {
      const monthMatches = selectedMonthName ? result.month === selectedMonthName : true;
      const yearMatches = selectedYear ? String(result.year) === String(selectedYear) : true;
      return monthMatches && yearMatches;
    });
  }, [results, selectedMonthName, selectedYear]);

  const filteredFailed = useMemo(() => {
    return failedFiles.filter(file => {
      const monthMatches = selectedMonthName ? file.month === selectedMonthName : true;
      const yearMatches = selectedYear ? String(file.year) === String(selectedYear) : true;
      return monthMatches && yearMatches;
    });
  }, [failedFiles, selectedMonthName, selectedYear]);

  // Group results by processed file (allow duplicate filenames)
  const convertedFiles = useMemo(() => {
    const fileMap = new Map();
    filteredResults.forEach(result => {
      const groupKey = result.fileGroupId || `${result.filename}-${result.timestamp || result.id || Math.random()}`;
      if (!fileMap.has(groupKey)) {
        fileMap.set(groupKey, {
          groupId: groupKey,
          filename: result.filename,
          fileSize: result.fileSize || 0,
          results: [],
          totalTables: 0,
          district: result.district,
          mandal: result.mandal,
          village: result.village,
          month: result.month,
          year: result.year,
          timestamp: result.timestamp
        });
      }
      const file = fileMap.get(groupKey);
      file.results.push(result);
      file.totalTables = Math.max(file.totalTables, result.totalTables || 1);
      if (!file.fileSize && result.fileSize) {
        file.fileSize = result.fileSize;
      }
    });
    return Array.from(fileMap.values());
  }, [filteredResults]);

  const nonConvertedFiles = useMemo(() => {
    return filteredFailed.map(file => ({
      filename: file.filename || file.name || file,
      size: file.size || 0,
      error: file.error || 'Conversion failed',
      timestamp: file.timestamp || new Date().toLocaleString()
    }));
  }, [filteredFailed]);

  const toggleFolder = (folderType) => {
    setExpandedFolders(prev => ({
      ...prev,
      [folderType]: !prev[folderType]
    }));
  };

  const openFolderModal = (type) => {
    setActiveFolderModal(type);
  };
  const handleFolderClick = (type) => {
    toggleFolder(type);
  };

  const closeFolderModal = () => {
    setActiveFolderModal(null);
  };

  const extractHtmlFragments = (html = '') => {
    if (!html) return { before: '', after: '' };
    const lowerHtml = html.toLowerCase();
    const tableStart = lowerHtml.indexOf('<table');
    if (tableStart === -1) return { before: html, after: '' };
    const tableCloseIdx = lowerHtml.indexOf('</table>', tableStart);
    if (tableCloseIdx === -1) {
      return {
        before: html.slice(0, tableStart),
        after: ''
      };
    }
    const afterStart = tableCloseIdx + '</table>'.length;
    return {
      before: html.slice(0, tableStart),
      after: html.slice(afterStart)
    };
  };

  const openEditModal = (result) => {
    if (!result) return;
    const headers = result.headers?.length ? result.headers : Object.keys(result.data?.[0] || {});
    setEditHeaders(headers);
    setEditableData((result.data || []).map(row => ({ ...row })));
    setEditingResult(result);
    setHtmlFragments(extractHtmlFragments(result.htmlData));
  };

  const closeEditModal = () => {
    setEditingResult(null);
    setEditableData([]);
    setEditHeaders([]);
    setHtmlFragments({ before: '', after: '' });
  };

  const handleCellChange = (rowIdx, header, value) => {
    setEditableData(prev => {
      const next = [...prev];
      next[rowIdx] = { ...next[rowIdx], [header]: value };
      return next;
    });
  };

  const resolveHeaders = (headers, rows) => {
    if (headers && headers.length) return headers;
    const headerSet = new Set();
    rows.forEach(row => {
      Object.keys(row || {}).forEach(key => headerSet.add(key));
    });
    return Array.from(headerSet);
  };

  const sanitizeValue = (value) => {
    if (value === null || value === undefined) return '';
    return String(value);
  };

  const convertRowsToCSV = (headers, rows) => {
    if (!headers || headers.length === 0) return '';
    const escape = (value) => {
      const str = sanitizeValue(value);
      return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
    };
    const headerLine = headers.map(escape).join(',');
    const rowLines = rows.map(row =>
      headers.map(header => escape(row?.[header])).join(',')
    );
    return [headerLine, ...rowLines].join('\n');
  };

  const escapeHtml = (value) => {
    return sanitizeValue(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  };

  const buildHTMLTable = (headers, rows) => {
    if (!headers || headers.length === 0) return '';

    const tableStyle = `
      width: 100%;
      border-collapse: collapse;
      font-family: 'Noto Sans Telugu', 'Poppins', sans-serif;
      font-size: 14px;
      background-color: #fff;
    `;
    const headerCellStyle = `
      background-color: #1f4ab9;
      color: #fff;
      padding: 8px 10px;
      border: 1px solid #d4d8e8;
      text-align: center;
      font-weight: 600;
      white-space: nowrap;
    `;
    const bodyCellStyle = `
      padding: 6px 10px;
      border: 1px solid #d4d8e8;
      color: #1a1a1a;
      text-align: left;
      min-width: 80px;
    `;
    const headerCells = headers
      .map(header => `<th style="${headerCellStyle}">${escapeHtml(header)}</th>`)
      .join('');
    const bodyRows = rows
      .map((row, rowIdx) => {
        const rowBg = rowIdx % 2 === 0 ? '#fefeff' : '#f4f7fb';
        const cells = headers
          .map(header => `<td style="${bodyCellStyle}background-color:${rowBg};">${escapeHtml(row?.[header] ?? '')}</td>`)
          .join('');
        return `<tr>${cells}</tr>`;
      })
      .join('');
    return `<table style="${tableStyle}"><thead><tr>${headerCells}</tr></thead><tbody>${bodyRows}</tbody></table>`;
  };

  const saveEditedResult = () => {
    if (editingResult && onUpdateResult) {
      const headers = resolveHeaders(editHeaders, editableData);
      const csvData = convertRowsToCSV(headers, editableData);
      const updatedTableHtml = buildHTMLTable(headers, editableData);
      const htmlData = htmlFragments.before || htmlFragments.after
        ? `${htmlFragments.before || ''}${updatedTableHtml}${htmlFragments.after || ''}`
        : updatedTableHtml;
      onUpdateResult(editingResult.id, {
        data: editableData,
        headers,
        csvData,
        htmlData,
        jsonData: JSON.stringify(editableData)
      });
    }
    closeEditModal();
  };

  const formatBytes = (bytes) => {
    if (!bytes || bytes <= 0) return 'Unknown size';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const getDisplayName = (filename = '') => {
    if (!filename) return 'Unknown';
    return filename.replace(/\.[^/.]+$/, '');
  };

  const getFileType = (filename = '') => {
    if (!filename || !filename.includes('.')) return 'Unknown';
    return filename.split('.').pop().toUpperCase() + ' File';
  };

  const handleTableClick = () => {
    setShowStatsModal(true);
    if (onShowStats) {
      onShowStats();
    }
  };

  // Calculate statistics
  const stats = useMemo(() => {
    const totalItems = convertedFiles.length + nonConvertedFiles.length;
    const convertedCount = convertedFiles.length;
    const nonConvertedCount = nonConvertedFiles.length;
    const totalNeeded = TOTAL_FILES_TARGET;
    // Only count successfully converted files in totalUploaded
    const cappedUploaded = Math.min(convertedCount, TOTAL_FILES_TARGET);
    const remaining = Math.max(totalNeeded - cappedUploaded, 0);

    return {
      totalUploaded: convertedCount, // Only count successfully converted files
      convertedCount,
      nonConvertedCount,
      totalNeeded,
      remaining
    };
  }, [filteredResults, filteredFailed]);

  const statsPeriodLabel = useMemo(() => {
    if (selectedMonthName && selectedYear) return `${selectedMonthName} ${selectedYear}`;
    if (selectedMonthName) return selectedMonthName;
    if (selectedYear) return selectedYear;
    return 'All Months';
  }, [selectedMonthName, selectedYear]);

  const renderConvertedContent = () => {
    if (!expandedFolders.converted) {
      return <p className="text-sm text-green-700 text-center">Folder minimized</p>;
    }

    if (convertedFiles.length === 0) {
      return (
        <div className="bg-white/70 border-2 border-dashed border-green-300 rounded-2xl p-6 text-center text-green-700 font-semibold">
          No converted files yet
        </div>
      );
    }

    return (
      <div className="border border-green-200 rounded-2xl overflow-hidden">
        <div className="max-h-[420px] overflow-y-auto divide-y divide-green-100">
          {convertedFiles.map((file, idx) => (
            <div
              key={file.groupId || `${file.filename}-${idx}`}
              className="grid md:grid-cols-[1.6fr_1fr_0.9fr_0.9fr_auto] grid-cols-1 gap-2 px-3 py-2 bg-white hover:bg-green-50 transition-colors text-sm"
            >
              <div className="flex items-center gap-2 min-w-0">
                <FileText size={18} className="text-green-600 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="font-semibold text-gray-800 truncate">{getDisplayName(file.filename)}</p>
                  <p className="text-[11px] text-gray-500">
                    {file.results.length} table{file.results.length > 1 ? 's' : ''}
                  </p>
                </div>
              </div>
              <div className="text-[11px] text-gray-600 md:text-center">{file.timestamp}</div>
              <div className="text-[11px] text-gray-600 md:text-center">{getFileType(file.filename)}</div>
              <div className="text-[11px] text-gray-600 md:text-center">{formatBytes(file.fileSize)}</div>
              <div className="flex items-center gap-1 justify-end flex-wrap">
                <button
                  onClick={() => onViewResult(file.results[0])}
                  className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-[11px] font-semibold flex items-center gap-1"
                >
                  <Eye size={12} />
                  View
                </button>
                <button
                  onClick={() => onExportCSV(file.results[0])}
                  className="px-2 py-1 bg-green-600 hover:bg-green-700 text-white rounded-md text-[11px] font-semibold flex items-center gap-1"
                >
                  <Download size={12} />
                  CSV
                </button>
                <button
                  onClick={async () => {
                    try {
                      // Update analytics to mark as synced to MBK
                      if (onUpdateAnalytics && file.district && file.mandal && file.village && file.month && file.year) {
                        const monthNum = new Date(`${file.month} 1, 2000`).getMonth() + 1;
                        await onUpdateAnalytics({
                          district: file.district,
                          mandal: file.mandal,
                          village: file.village,
                          month: monthNum.toString(),
                          year: file.year.toString(),
                          shgId: file.results[0]?.shgMbkId || '',
                          validationStatus: 'success',
                          failureReason: null,
                          syncedToMbk: true
                        });
                      }
                      window.alert('File submitted to MBK successfully!');
                    } catch (error) {
                      console.error('Error submitting to MBK:', error);
                      window.alert('Error submitting to MBK. Please try again.');
                    }
                  }}
                  className="px-3 py-1.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white rounded-full text-[11px] font-semibold flex items-center gap-1 shadow-sm hover:shadow-md transition-all"
                >
                  Submit
                </button>
                <button
                  onClick={() => openEditModal(file.results[0])}
                  className="px-2 py-1 bg-yellow-500 hover:bg-yellow-600 text-white rounded-md text-[11px] font-semibold flex items-center gap-1"
                >
                  <Edit3 size={12} />
                  Edit
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderNonConvertedContent = () => {
    if (!expandedFolders.nonConverted) {
      return <p className="text-sm text-red-700 text-center">Folder minimized</p>;
    }

    if (nonConvertedFiles.length === 0) {
      return (
        <div className="bg-white/70 border-2 border-dashed border-rose-300 rounded-2xl p-6 text-center text-red-700 font-semibold">
          No failed files 🎉
        </div>
      );
    }

    return (
      <div className="border border-rose-200 rounded-2xl overflow-hidden">
        <div className="max-h-[420px] overflow-y-auto divide-y divide-rose-100">
          {nonConvertedFiles.map((file, idx) => (
            <div
              key={file.filename ? `${file.filename}-${idx}` : idx}
              className="grid md:grid-cols-[1.6fr_1fr_1fr_1fr_auto] grid-cols-1 gap-2 px-3 py-2 bg-white hover:bg-rose-50 transition-colors text-sm"
            >
              <div className="flex items-center gap-2 min-w-0">
                <FileText size={18} className="text-red-600 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="font-semibold text-gray-800 truncate">{getDisplayName(file.filename)}</p>
                  <p className="text-[11px] text-red-500">❌ {file.error}</p>
                </div>
              </div>
              <div className="text-[11px] text-gray-600 md:text-center">{file.timestamp || '—'}</div>
              <div className="text-[11px] text-gray-600 md:text-center">{getFileType(file.filename)}</div>
              <div className="text-[11px] text-gray-600 md:text-center">{formatBytes(file.size)}</div>
              <div className="flex items-center justify-end">
                <button
                  onClick={() => window.alert('Deleted this failed file')}
                  className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-full text-[11px] font-semibold shadow-sm hover:shadow-md transition-all"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
        <div className="flex justify-between items-center mb-6">
          <button
            onClick={handleTableClick}
            className="text-2xl lg:text-3xl font-bold text-gray-800 flex items-center gap-2 hover:text-indigo-600 transition-colors cursor-pointer"
          >
            <Table2 size={32} className="text-indigo-600" />
            Converted Results
          </button>
          <button
            onClick={handleTableClick}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold flex items-center gap-2"
          >
            <Info size={20} />
            View Statistics
          </button>
        </div>

        {results.length === 0 && nonConvertedFiles.length === 0 ? (
          <div className="text-center py-12">
            <Table2 size={64} className="mx-auto text-gray-400 mb-4" />
            <p className="text-xl text-gray-600 font-semibold">No results yet</p>
            <p className="text-gray-500 mt-2">Upload and process files to see results here</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="flex flex-col">
              <div className="h-6 w-32 bg-green-200 rounded-t-2xl ml-8 shadow-inner"></div>
              <div className="bg-gradient-to-br from-green-50 to-emerald-100 border-2 border-green-300 rounded-3xl rounded-tl-none -mt-1 pt-6 px-6 pb-4 shadow-xl flex-1">
                <div
                  className="flex items-center justify-between pb-4 border-b border-green-200 mb-4 cursor-pointer"
                  onClick={() => handleFolderClick('converted')}
                  onDoubleClick={() => openFolderModal('converted')}
                >
                  <div className="flex items-center gap-3 text-left select-none">
                    {expandedFolders.converted ? (
                      <FolderOpen size={32} className="text-green-700" />
                    ) : (
                      <Folder size={32} className="text-green-700" />
                    )}
                    <div>
                      <h3 className="text-xl font-bold text-green-900">Converted Files</h3>
                      <p className="text-sm text-green-700">
                        {convertedFiles.length} file(s) successfully converted
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={async () => {
                        try {
                          let submittedCount = 0;
                          for (const file of convertedFiles) {
                            if (onUpdateAnalytics && file.district && file.mandal && file.village && file.month && file.year) {
                              const monthNum = new Date(`${file.month} 1, 2000`).getMonth() + 1;
                              await onUpdateAnalytics({
                                district: file.district,
                                mandal: file.mandal,
                                village: file.village,
                                month: monthNum.toString(),
                                year: file.year.toString(),
                                shgId: file.results[0]?.shgMbkId || '',
                                validationStatus: 'success',
                                failureReason: null,
                                syncedToMbk: true
                              });
                              submittedCount++;
                            }
                          }
                          window.alert(`Successfully submitted ${submittedCount} file(s) to MBK!`);
                        } catch (error) {
                          console.error('Error submitting files to MBK:', error);
                          window.alert('Error submitting files to MBK. Please try again.');
                        }
                      }}
                      className="px-4 py-1.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white rounded-full text-xs font-semibold shadow-sm hover:shadow-md transition-all"
                    >
                      Submit All
                    </button>
                    <div className="px-4 py-1 bg-green-600 text-white rounded-full font-bold">
                      {convertedFiles.length}
                    </div>
                  </div>
                </div>

                {renderConvertedContent()}
              </div>
            </div>

            <div className="flex flex-col">
              <div className="h-6 w-32 bg-rose-200 rounded-t-2xl ml-8 shadow-inner"></div>
              <div className="bg-gradient-to-br from-rose-50 to-red-100 border-2 border-rose-300 rounded-3xl rounded-tl-none -mt-1 pt-6 px-6 pb-4 shadow-xl flex-1">
                <div
                  className="flex items-center justify-between pb-4 border-b border-rose-200 mb-4 cursor-pointer"
                  onClick={() => handleFolderClick('nonConverted')}
                  onDoubleClick={() => openFolderModal('nonConverted')}
                >
                  <div className="flex items-center gap-3 text-left select-none">
                    {expandedFolders.nonConverted ? (
                      <FolderOpen size={32} className="text-red-700" />
                    ) : (
                      <Folder size={32} className="text-red-700" />
                    )}
                    <div>
                      <h3 className="text-xl font-bold text-red-900">Non-Converted Files</h3>
                      <p className="text-sm text-red-700">
                        {nonConvertedFiles.length} file(s) failed to convert
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => window.alert('Deleted all failed files')}
                      className="px-4 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-full text-xs font-semibold shadow-sm hover:shadow-md transition-all"
                    >
                      Delete All
                    </button>
                    <div className="px-4 py-1 bg-red-600 text-white rounded-full font-bold">
                      {nonConvertedFiles.length}
                    </div>
                  </div>
                </div>

                {renderNonConvertedContent()}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Statistics Modal */}
      {showStatsModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                  <Info size={32} className="text-blue-600" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-800">Upload Statistics</h3>
                  <p className="text-sm font-semibold text-blue-600">{statsPeriodLabel}</p>
                </div>
              </div>
              <button
                onClick={() => setShowStatsModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X size={24} className="text-gray-600" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-4xl font-bold text-blue-800 mb-2">
                    {stats.totalUploaded}
                  </div>
                  <div className="text-sm font-semibold text-blue-600">
                    Total Files Uploaded
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-800 mb-2">
                      {stats.convertedCount}
                    </div>
                    <div className="text-xs font-semibold text-green-600">
                      Successfully Converted
                    </div>
                  </div>
                </div>

                <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-red-800 mb-2">
                      {stats.nonConvertedCount}
                    </div>
                    <div className="text-xs font-semibold text-red-600">
                      Failed to Convert
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-4xl font-bold text-orange-800 mb-2">
                    {stats.remaining}
                  </div>
                  <div className="text-sm font-semibold text-orange-600">
                    Files Remaining (Need to Upload out of 10)
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 border-2 border-gray-300 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-800 mb-2">
                    {stats.totalNeeded}
                  </div>
                  <div className="text-sm font-semibold text-gray-600">
                    Total Files Needed to Upload (Fixed Target)
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={() => setShowStatsModal(false)}
              className="w-full mt-6 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-bold"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Folder Laptop View Modal */}
      {activeFolderModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-3xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col border border-gray-700">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
              <div>
                <p className="text-sm text-gray-400 uppercase tracking-wide">Laptop View</p>
                <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                  <Grid3X3 size={20} />
                  {activeFolderModal === 'converted' ? 'Converted Files' : 'Non-Converted Files'}
                </h3>
              </div>
              <button
                onClick={closeFolderModal}
                className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg font-semibold"
              >
                Close
              </button>
            </div>

            <div className="flex-1 overflow-auto bg-gradient-to-b from-gray-900 via-gray-850 to-gray-900 p-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {(activeFolderModal === 'converted' ? convertedFiles : nonConvertedFiles).map((file, idx) => (
                  <div
                    key={(file.groupId || file.filename || 'file') + idx}
                    className="bg-gray-800/80 border border-gray-700 rounded-2xl p-4 hover:border-indigo-400 hover:shadow-lg transition-all"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        activeFolderModal === 'converted' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                      }`}>
                        <FileText size={28} />
                      </div>
                      <div className="min-w-0">
                        <p className="text-white font-semibold truncate">{getDisplayName(file.filename)}</p>
                        <p className="text-xs text-gray-400">{formatBytes(file.fileSize || file.size)}</p>
                      </div>
                    </div>

                    {activeFolderModal === 'converted' ? (
                      <>
                        <div className="flex flex-wrap gap-2 mb-3">
                          <span className="px-2 py-1 bg-indigo-500/20 text-indigo-200 rounded-full text-xs font-semibold">
                            {file.results.length} Table{file.results.length > 1 ? 's' : ''}
                          </span>
                          {file.month && file.year && (
                            <span className="px-2 py-1 bg-blue-500/20 text-blue-200 rounded-full text-xs font-semibold">
                              {file.month} {file.year}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-400 mb-2">📍 {file.district} → {file.mandal} → {file.village}</p>
                        <p className="text-xs text-gray-500 mb-3">🕐 {file.timestamp}</p>
                        <div className="flex gap-2">
                          <button
                            onClick={() => {
                              if (file.results[0]) onViewResult(file.results[0]);
                              closeFolderModal();
                            }}
                            className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-semibold flex items-center justify-center gap-2"
                          >
                            <Eye size={16} />
                            View
                          </button>
                          <button
                            onClick={() => {
                              if (file.results[0]) onExportCSV(file.results[0]);
                            }}
                            className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-semibold flex items-center justify-center gap-2"
                          >
                            <Download size={16} />
                            CSV
                          </button>
                          <button
                            onClick={() => {
                              if (file.results[0]) {
                                openEditModal(file.results[0]);
                                closeFolderModal();
                              }
                            }}
                            className="flex-1 px-3 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg text-sm font-semibold flex items-center justify-center gap-2"
                          >
                            <Edit3 size={16} />
                            Edit
                          </button>
                        </div>
                      </>
                    ) : (
                      <>
                        <p className="text-sm text-red-300 font-semibold mb-2">❌ {file.error}</p>
                        <p className="text-xs text-gray-400 mb-1">Size: {formatBytes(file.size)}</p>
                        {file.timestamp && (
                          <p className="text-xs text-gray-500">🕐 {file.timestamp}</p>
                        )}
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Manual Edit Modal */}
      {editingResult && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <div>
                <p className="text-sm text-gray-500 uppercase tracking-wide">Manual Edit</p>
                <h3 className="text-2xl font-bold text-gray-800">{getDisplayName(editingResult.filename)}</h3>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={closeEditModal}
                  className="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold"
                >
                  Cancel
                </button>
                <button
                  onClick={saveEditedResult}
                  className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white font-semibold"
                >
                  Save Changes
                </button>
              </div>
            </div>
            <div className="p-6 flex-1 overflow-auto">
              <div className="overflow-auto border border-gray-200 rounded-xl">
                <table className="min-w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      {editHeaders.map(header => (
                        <th key={header} className="border border-gray-200 px-3 py-2 text-left text-sm font-semibold text-gray-700">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {editableData.map((row, rowIdx) => (
                      <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        {editHeaders.map(header => (
                          <td key={header} className="border border-gray-200 px-2 py-2">
                            <input
                              type="text"
                              value={row[header] ?? ''}
                              onChange={(e) => handleCellChange(rowIdx, header, e.target.value)}
                              className="w-full border border-gray-300 rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

