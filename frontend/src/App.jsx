import React, { useState, useEffect, useRef } from 'react';
import { Download, Eye, FileText, Table2, CheckCircle, AlertCircle, X, Upload, MapPin, RotateCw, RotateCcw, CheckSquare, ZoomIn, ZoomOut, BarChart3, TrendingUp } from 'lucide-react';
import ConvertedResults from './ConvertedResults';
import DataAnalytics from './DataAnalytics';
import FinancialAnalytics from './FinancialAnalytics';

const STORAGE_KEY_RESULTS = 'ocr_results_v1';
const STORAGE_KEY_FAILED = 'ocr_failed_v1';

const sanitizeValue = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' && !Number.isFinite(value)) return '';
  return String(value);
};

const convertRowsToCSV = (headers = [], rows = []) => {
  if (!headers.length) return '';
  const escapeValue = (value) => {
    const str = sanitizeValue(value);
    return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
  };
  const headerLine = headers.map(escapeValue).join(',');
  const rowLines = rows.map((row) =>
    headers.map((header) => escapeValue(row?.[header] ?? '')).join(',')
  );
  return [headerLine, ...rowLines].join('\n');
};

const escapeHtml = (value) =>
  sanitizeValue(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');

const buildHTMLTable = (headers = [], rows = []) => {
  if (!headers.length) return '';
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
  const bodyCellBaseStyle = `
    padding: 6px 10px;
    border: 1px solid #d4d8e8;
    color: #1a1a1a;
    text-align: left;
    min-width: 80px;
  `;
  const headerCells = headers
    .map((header) => `<th style="${headerCellStyle}">${escapeHtml(header)}</th>`)
    .join('');
  const bodyRows = rows
    .map((row, rowIdx) => {
      const rowBg = rowIdx % 2 === 0 ? '#fefeff' : '#f4f7fb';
      const cells = headers
        .map(
          (header) =>
            `<td style="${bodyCellBaseStyle}background-color:${rowBg};">${escapeHtml(
              row?.[header] ?? ''
            )}</td>`
        )
        .join('');
      return `<tr>${cells}</tr>`;
    })
    .join('');
  return `<table style="${tableStyle}"><thead><tr>${headerCells}</tr></thead><tbody>${bodyRows}</tbody></table>`;
};

const normalizeStructuredTableData = (tableData = {}) => {
  const baseHeaders = Array.isArray(tableData.column_headers)
    ? tableData.column_headers.map((header, idx) => {
        if (!header) return `Column ${idx + 1}`;
        return (
          (header.label && header.label.trim()) ||
          header.key ||
          `Column ${typeof header.index === 'number' ? header.index + 1 : idx + 1}`
        );
      })
    : [];
  const headers = [...baseHeaders];
  const addHeader = (label) => {
    if (!label) return;
    if (!headers.includes(label)) {
      headers.push(label);
    }
  };

  const rows = [];
  const dataRows = Array.isArray(tableData.data_rows) ? tableData.data_rows : [];

  dataRows.forEach((row) => {
    const rowObj = {};

    if (Array.isArray(row?.cells) && row.cells.length) {
      row.cells.forEach((cell) => {
        const colIndex =
          typeof cell?.col_index === 'number' && cell.col_index >= 0 ? cell.col_index : -1;
        const fallbackLabel =
          (colIndex >= 0 && headers[colIndex]) ||
          cell?.key ||
          `Column ${colIndex >= 0 ? colIndex + 1 : headers.length + 1}`;
        const label = (cell?.label && cell.label.trim()) || fallbackLabel;
        addHeader(label);
        rowObj[label] = sanitizeValue(cell?.text);
      });
    }

    Object.keys(row || {}).forEach((key) => {
      if (['cells', 'confidence', 'row_number', 'row_index'].includes(key)) return;
      const label = key;
      addHeader(label);
      rowObj[label] = sanitizeValue(row[key]);
    });

    rows.push(rowObj);
  });

  const filteredHeaders = headers.filter(Boolean);
  const csv = convertRowsToCSV(filteredHeaders, rows);
  const html = buildHTMLTable(filteredHeaders, rows);

  return {
    dataframe: rows,
    row_count: rows.length,
    col_count: filteredHeaders.length,
    csv,
    html,
    json: JSON.stringify(rows),
    headers: filteredHeaders,
    metadata: {
      shg_mbk_id: tableData.shg_mbk_id || '',
      total_columns: tableData.total_columns || filteredHeaders.length,
      total_rows: tableData.total_rows || rows.length,
    },
  };
};

const normalizeExtractionResponse = (responseData) => {
  if (!responseData || typeof responseData !== 'object') return [];

  if (Array.isArray(responseData.tables)) {
    return responseData.tables;
  }

  const normalized = [];
  const files = Array.isArray(responseData.files) ? responseData.files : [];

  files.forEach((fileEntry = {}) => {
    const pages = Array.isArray(fileEntry.pages) ? fileEntry.pages : [];
    pages.forEach((pageEntry = {}, pageIdx) => {
      if (pageEntry.success && pageEntry.table_data) {
        const normalizedTable = normalizeStructuredTableData(pageEntry.table_data);
        if (normalizedTable.dataframe.length > 0) {
          normalized.push({
            ...normalizedTable,
            source: {
              filename: fileEntry.filename,
              fileType: fileEntry.file_type,
              pageNumber: pageEntry.page ?? pageIdx + 1,
              totalPages: pageEntry.total_pages ?? pages.length,
            },
          });
        }
      }
    });
  });

  return normalized;
};

export default function EnhancedTableOCRSystem() {
  const [files, setFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [toasts, setToasts] = useState([]);
  const [results, setResults] = useState([]);
  const [failedFiles, setFailedFiles] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [previewFile, setPreviewFile] = useState(null);
  const [previewZoom, setPreviewZoom] = useState(1);
  const [resultZoom, setResultZoom] = useState(1);
  const [uploadStats, setUploadStats] = useState({ total: 0, validated: 0, pending: 0 });
  const [showDuplicateModal, setShowDuplicateModal] = useState(false);
  const [duplicateFiles, setDuplicateFiles] = useState([]);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [duplicateResultFiles, setDuplicateResultFiles] = useState([]);
  const [showDuplicateUploadModal, setShowDuplicateUploadModal] = useState(false);
  const [duplicateUploadFiles, setDuplicateUploadFiles] = useState([]);
  const [showDuplicateResultModal, setShowDuplicateResultModal] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  
  const [districts, setDistricts] = useState([]);
  const [mandals, setMandals] = useState([]);
  const [villages, setVillages] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState("");
  const [selectedMandal, setSelectedMandal] = useState("");
  const [selectedVillage, setSelectedVillage] = useState("");
  const [selectedMonth, setSelectedMonth] = useState("");
  const [selectedYear, setSelectedYear] = useState("");

  const fileInputRef = useRef(null);
  const toastTimers = useRef({});

  const API_BASE = 'http://localhost:5002/OCR';
  const MAX_FILE_SIZE = 16 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'pdf', 'tiff', 'tif', 'bmp', 'webp'];

  // Auto-load CSV file & persisted data on component mount
  useEffect(() => {
    fetchLocations();
    const interval = setInterval(() => {
      fetchLocations();
    }, 300000);
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    return () => {
      Object.values(toastTimers.current || {}).forEach(timerId => clearTimeout(timerId));
      toastTimers.current = {};
    };
  }, []);

  useEffect(() => {
    localStorage.removeItem(STORAGE_KEY_RESULTS);
    localStorage.removeItem(STORAGE_KEY_FAILED);
  }, []);

  async function fetchLocations() {
    try {
      // Detect base path - use Vite's BASE_URL if available, otherwise detect from URL
      let basePath = '';
      try {
        // Vite provides import.meta.env.BASE_URL (e.g., '/OCR/')
        if (typeof import.meta !== 'undefined' && import.meta.env?.BASE_URL) {
          basePath = import.meta.env.BASE_URL.replace(/\/$/, ''); // Remove trailing slash
        }
      } catch (e) {
        // Fallback: detect from current path
        const currentPath = window.location.pathname;
        if (currentPath.startsWith('/OCR')) {
          basePath = '/OCR';
        } else {
          const pathParts = currentPath.split('/').filter(p => p);
          if (pathParts.length > 0) {
            basePath = '/' + pathParts[0];
          }
        }
      }
      
      // Build possible paths - prioritize base path for GitHub Pages
      const possiblePaths = [
        // Base path first (for GitHub Pages deployment with /OCR/ base)
        ...(basePath ? [`${basePath}/districts-mandals.csv`] : []),
        // Relative paths (work in local dev)
        "./districts-mandals.csv",
        "districts-mandals.csv",
        // Absolute root path
        "/districts-mandals.csv",
        // Fallback patterns
        "/OCR/districts-mandals.csv",
        "OCR/districts-mandals.csv"
      ];
      
      // Remove duplicates while preserving order
      const uniquePaths = Array.from(new Set(possiblePaths));
      
      let csvText = null;
      let lastError = null;
      let successfulPath = null;
      
      for (const path of uniquePaths) {
        try {
          const res = await fetch(`${path}?t=${Date.now()}`);
          if (res.ok) {
            const contentType = res.headers.get('content-type') || '';
            const text = await res.text();
            
            // Validate that we got CSV content, not HTML
            if (text.trim().toLowerCase().startsWith('<!doctype') || 
                text.trim().toLowerCase().startsWith('<html')) {
              console.warn(`Path ${path} returned HTML instead of CSV, trying next path...`);
              continue;
            }
            
            // Check if it looks like CSV (has comma-separated values)
            if (text.includes(',') || text.includes('mandal') || text.includes('district')) {
              csvText = text;
              successfulPath = path;
              console.log(`Successfully loaded CSV from: ${path}`);
              break;
            } else {
              console.warn(`Path ${path} doesn't appear to be CSV content, trying next path...`);
              continue;
            }
          }
        } catch (err) {
          lastError = err;
          continue;
        }
      }
      
      if (!csvText) {
        throw new Error(`Failed to load CSV from any path. Last error: ${lastError?.message || 'Unknown'}`);
      }
      
      const lines = csvText.split('\n').filter(line => line.trim());
      console.log(`CSV loaded: ${lines.length} lines from ${successfulPath}`);
      if (lines.length > 0) {
        console.log(`First line: ${lines[0]}`);
      }
      await loadCSVDataFromText(lines);
    } catch (err) {
      console.warn("Could not auto-load CSV file:", err.message);
    }
  }

  const showToast = (type, title, details) => {
    const id = `${Date.now()}-${Math.random()}`;
    const detailLines = Array.isArray(details) ? details : [details];
    setToasts(prev => [...prev, { id, type, title, details: detailLines }]);
    const timeoutId = setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
      delete toastTimers.current[id];
    }, 4500);
    toastTimers.current[id] = timeoutId;
  };

  const dismissToast = (id) => {
    if (toastTimers.current[id]) {
      clearTimeout(toastTimers.current[id]);
      delete toastTimers.current[id];
    }
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  const loadCSVDataFromText = async (lines) => {
    try {
      if (lines.length < 2) {
        console.warn('CSV file has insufficient lines:', lines.length);
        return;
      }

      const headerValues = [];
      let current = '';
      let inQuotes = false;
      for (let char of lines[0]) {
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          headerValues.push(current.trim().replace(/^"|"$/g, '').toLowerCase());
          current = '';
        } else {
          current += char;
        }
      }
      headerValues.push(current.trim().replace(/^"|"$/g, '').toLowerCase());
      
      // Debug: log what headers were found
      console.log('CSV Headers found:', headerValues);
      
      // More flexible matching - check for both singular and plural forms
      const mandalIdx = headerValues.findIndex(h => 
        h.includes('mandal') || h === 'mandal' || h === 'mandals'
      );
      const districtIdx = headerValues.findIndex(h => 
        h.includes('district') || h === 'district' || h === 'districts'
      );
      const villageIdx = headerValues.findIndex(h => 
        h.includes('village') || h === 'village' || h === 'villages'
      );
      
      if (districtIdx === -1 || mandalIdx === -1) {
        console.error('CSV must contain mandals and districts columns');
        console.error('Available headers:', headerValues);
        console.error('First line of CSV:', lines[0]);
        return;
      }

      const districtMap = new Map();
      
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = [];
        let current = '';
        let inQuotes = false;
        
        for (let char of line) {
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            values.push(current.trim().replace(/^"|"$/g, ''));
            current = '';
          } else {
            current += char;
          }
        }
        values.push(current.trim().replace(/^"|"$/g, ''));
        
        const mandalName = values[mandalIdx]?.trim();
        const districtName = values[districtIdx]?.trim();
        const villageName = villageIdx !== -1 ? values[villageIdx]?.trim() : '';
        
        if (!districtName || !mandalName) continue;
        
        if (!districtMap.has(districtName)) {
          districtMap.set(districtName, {
            id: `d_${districtName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`,
            name: districtName,
            mandals: new Map()
          });
        }
        
        const district = districtMap.get(districtName);
        const mandalKey = mandalName.toLowerCase();
        
        if (!district.mandals.has(mandalKey)) {
          district.mandals.set(mandalKey, {
            id: `m_${mandalName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`,
            name: mandalName,
            villages: []
          });
        }
        
        const mandal = district.mandals.get(mandalKey);
        
        if (villageName) {
          const villageId = `v_${villageName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`;
          if (!mandal.villages.some(v => v.id === villageId)) {
            mandal.villages.push({
              id: villageId,
              name: villageName
            });
          }
        } else if (mandal.villages.length === 0) {
          // If no villages specified, add mandal name as default village
          const defaultVillageId = `v_${mandalName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`;
          if (!mandal.villages.some(v => v.id === defaultVillageId)) {
            mandal.villages.push({
              id: defaultVillageId,
              name: mandalName
            });
          }
        }
      }
      
      const districtsArray = Array.from(districtMap.values()).map(d => ({
        ...d,
        mandals: Array.from(d.mandals.values())
          .map(m => ({
            ...m,
            villages: m.villages.sort((a, b) => a.name.localeCompare(b.name))
          }))
          .sort((a, b) => a.name.localeCompare(b.name))
      })).sort((a, b) => a.name.localeCompare(b.name));
      
      setDistricts(districtsArray);
      
      const totalMandals = districtsArray.reduce((sum, d) => sum + d.mandals.length, 0);
      const totalVillages = districtsArray.reduce((sum, d) => 
        sum + d.mandals.reduce((mSum, m) => mSum + m.villages.length, 0), 0
      );
      
      console.log('✅ CSV Data loaded successfully:');
      console.log(`📊 Total Districts: ${districtsArray.length}`);
      console.log(`📊 Total Mandals: ${totalMandals}`);
      console.log(`📊 Total Villages: ${totalVillages}`);
      
      // Debug: Show first few entries
      if (districtsArray.length > 0) {
        console.log('\n📋 Sample Data:');
        districtsArray.slice(0, 2).forEach(d => {
          console.log(`\n  District: ${d.name}`);
          d.mandals.slice(0, 3).forEach(m => {
            console.log(`    - Mandal: ${m.name}`);
            m.villages.slice(0, 3).forEach(v => {
              console.log(`      • Village: ${v.name}`);
            });
            if (m.villages.length > 3) {
              console.log(`      ... and ${m.villages.length - 3} more villages`);
            }
          });
        });
      }
    } catch (error) {
      console.error('Error loading CSV:', error);
    }
  };

  useEffect(() => {
    const validated = files.filter(f => f.validated).length;
    const pending = files.length - validated;
    setUploadStats({ total: files.length, validated, pending });
  }, [files]);

  useEffect(() => {
    if (!selectedDistrict) {
      setMandals([]);
      setVillages([]);
      setSelectedMandal("");
      setSelectedVillage("");
      return;
    }
    const district = districts.find(d => d.id === selectedDistrict);
    if (district) {
      setMandals(district.mandals || []);
      setSelectedMandal("");
      setSelectedVillage("");
      setVillages([]);
    }
  }, [selectedDistrict, districts]);

  useEffect(() => {
    if (!selectedMandal) {
      setVillages([]);
      setSelectedVillage("");
      return;
    }
    const mandal = mandals.find(m => m.id === selectedMandal);
    if (mandal) {
      setVillages(mandal.villages || []);
      setSelectedVillage("");
    }
  }, [selectedMandal, mandals]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    const duplicates = [];
    const invalidFiles = [];
    const oversizedFiles = [];
    const validNewFiles = [];

    selectedFiles.forEach(file => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      
      if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
        invalidFiles.push(file.name);
        return;
      }

      if (file.size > MAX_FILE_SIZE) {
        oversizedFiles.push(file.name);
        return;
      }

      const isDuplicate = files.some(f => 
        f.name === file.name && f.size === file.size
      );

      if (isDuplicate) {
        duplicates.push(file.name);
      } else {
        validNewFiles.push(file);
      }
    });

    const currentMonthName = selectedMonth
      ? new Date(2000, parseInt(selectedMonth) - 1).toLocaleString('default', { month: 'long' })
      : null;
    const currentYearValue = selectedYear ? String(selectedYear) : null;

    const duplicateMonthlyFiles = [];
    const filteredValidNewFiles = validNewFiles.filter(file => {
      if (!currentMonthName || !currentYearValue) return true;
      const duplicateKey = `${file.name}-${currentMonthName}-${currentYearValue}`;
      const exists = results.some(r => r.duplicateKey === duplicateKey);
      if (exists) {
        duplicateMonthlyFiles.push(file.name);
        return false;
      }
      return true;
    });

    const newFiles = filteredValidNewFiles.map(file => ({
      id: Date.now() + Math.random(),
      file: file,
      name: file.name,
      size: file.size,
      type: file.type,
      validated: false,
      previewUrl: null,
      rotation: 0
    }));

    let messages = [];
    if (newFiles.length > 0) {
      messages.push(`✅ ${newFiles.length} file(s) added successfully`);
      showToast('success', 'Files ready to upload', [
        `${newFiles.length} file${newFiles.length > 1 ? 's' : ''} added to the queue`
      ]);
    }
    if (duplicates.length > 0) {
      setDuplicateFiles(duplicates);
      setShowDuplicateModal(true);
      const duplicatePreview = duplicates.length > 3 
        ? [...duplicates.slice(0, 3), `...and ${duplicates.length - 3} more`]
        : duplicates;
      showToast('warning', 'Duplicate file name(s) skipped', duplicatePreview);
    }
    if (invalidFiles.length > 0) {
      messages.push(`❌ ${invalidFiles.length} invalid file type(s)`);
    }
    if (oversizedFiles.length > 0) {
      messages.push(`❌ ${oversizedFiles.length} file(s) too large (max 16MB)`);
    }

    if (messages.length > 0) {
      setMessage(messages.join('\n'));
      setMessageType(newFiles.length > 0 ? 'info' : 'error');
    }

    if (newFiles.length > 0) {
      setFiles([...files, ...newFiles]);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    if (duplicateMonthlyFiles.length > 0) {
      setDuplicateUploadFiles(duplicateMonthlyFiles);
      setShowDuplicateUploadModal(true);
      const monthlyPreview = duplicateMonthlyFiles.length > 3
        ? [...duplicateMonthlyFiles.slice(0, 3), `...and ${duplicateMonthlyFiles.length - 3} more`]
        : duplicateMonthlyFiles;
      showToast('warning', 'Already uploaded for this month', monthlyPreview);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const droppedFiles = Array.from(e.dataTransfer.files);
    const fakeEvent = { target: { files: droppedFiles } };
    handleFileChange(fakeEvent);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const removeFile = (fileId) => {
    const fileToRemove = files.find(f => f.id === fileId);
    if (fileToRemove?.previewUrl) {
      URL.revokeObjectURL(fileToRemove.previewUrl);
    }
    setFiles(files.filter(f => f.id !== fileId));
  };

  const validateFile = (fileId) => {
    setFiles(files.map(f => 
      f.id === fileId ? { ...f, validated: true } : f
    ));
    setMessage('✅ File validated successfully');
    setMessageType('success');
  };

  const validateAllFiles = () => {
    if (files.length === 0) {
      setMessage('❌ No files to validate');
      setMessageType('error');
      return;
    }
    setFiles(files.map(f => ({ ...f, validated: true })));
    setMessage(`✅ All ${files.length} file(s) validated successfully!`);
    setMessageType('success');
  };

  const previewFileHandler = (fileObj) => {
    if (!fileObj.previewUrl && fileObj.file) {
      fileObj.previewUrl = URL.createObjectURL(fileObj.file);
    }
    if (fileObj.rotation === undefined) {
      fileObj.rotation = 0;
    }
    setPreviewFile({ ...fileObj });
  };

  const closePreview = () => {
    setPreviewFile(null);
    setPreviewZoom(1);
  };

  const rotateImage = (direction) => {
    if (!previewFile) return;
    
    const rotationIncrement = direction === 'right' ? 90 : -90;
    const newRotation = ((previewFile.rotation || 0) + rotationIncrement) % 360;
    
    const updatedFile = { ...previewFile, rotation: newRotation };
    setPreviewFile(updatedFile);
    
    setFiles(files.map(f => 
      f.id === previewFile.id ? { ...f, rotation: newRotation } : f
    ));
  };

  const rotateImageFile = async (file, rotation) => {
    if (!rotation || rotation === 0 || rotation % 360 === 0) {
      return file;
    }

    try {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          img.onload = () => {
            if (rotation === 90 || rotation === 270 || rotation === -90 || rotation === -270) {
              canvas.width = img.height;
              canvas.height = img.width;
            } else {
              canvas.width = img.width;
              canvas.height = img.height;
            }
            
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.rotate((rotation * Math.PI) / 180);
            ctx.drawImage(img, -img.width / 2, -img.height / 2);
            
            canvas.toBlob((blob) => {
              if (blob) {
                const rotatedFile = new File([blob], file.name, {
                  type: file.type || 'image/jpeg',
                  lastModified: Date.now()
                });
                resolve(rotatedFile);
              } else {
                reject(new Error('Failed to create rotated image'));
              }
            }, file.type || 'image/jpeg', 0.95);
          };
          img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    } catch (error) {
      console.error('Error rotating image:', error);
      return file;
    }
  };

  // Check backend connection
  const checkBackendConnection = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout
      
      const response = await fetch(`${API_BASE}/api/health`, {
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      return response.ok;
    } catch (err) {
      return false;
    }
  };

  const handleProcess = async () => {
    if (files.length === 0) {
      setMessage('❌ Please select at least one file');
      setMessageType('error');
      return;
    }

    const allValidated = files.every(f => f.validated);
    if (!allValidated) {
      setMessage('❌ Please validate all files before processing');
      setMessageType('error');
      return;
    }

    if (!selectedDistrict || !selectedMandal || !selectedVillage) {
      setMessage('❌ Please select District, Mandal, and Village');
      setMessageType('error');
      return;
    }

    if (!selectedMonth || !selectedYear) {
      setMessage('❌ Please select Month and Year');
      setMessageType('error');
      return;
    }

    // Check backend connection before processing
    setMessage('🔍 Checking backend connection...');
    setMessageType('info');
    const isBackendAvailable = await checkBackendConnection();
    
    if (!isBackendAvailable) {
      setMessage(
        `❌ Backend Server Not Available\n\n` +
        `Cannot connect to backend server at ${API_BASE}\n\n` +
        `Please start the backend server:\n` +
        `1. Open a terminal/command prompt\n` +
        `2. Navigate to the backend folder\n` +
        `3. Run: python app.py\n\n` +
        `Or use: npm run dev:all (to start both frontend and backend)`
      );
      setMessageType('error');
      setProcessing(false);
      return;
    }

    try {
      setProcessing(true);
      setMessage('🔍 Extracting tables from images...');
      setMessageType('info');

      const allResults = [];
      const newFailedFiles = [];
    const resultDuplicates = [];
      const districtName = districts.find(d => d.id === selectedDistrict)?.name;
      const mandalName = mandals.find(m => m.id === selectedMandal)?.name;
      const villageName = villages.find(v => v.id === selectedVillage)?.name;
      const monthName = new Date(2000, parseInt(selectedMonth) - 1).toLocaleString('default', { month: 'long' });

      let totalProcessed = 0;
      let successCount = 0;
      let errorCount = 0;

      for (const fileObj of files) {
        totalProcessed++;
        setMessage(
          `🔬 Processing ${totalProcessed}/${files.length}: ${fileObj.name}\n` +
          `📐 Detecting table structure...`
        );

        let fileToProcess = fileObj.file;
        if (fileObj.rotation && fileObj.rotation !== 0) {
          try {
            fileToProcess = await rotateImageFile(fileObj.file, fileObj.rotation);
            setMessage(
              `🔄 Rotating image (${fileObj.rotation}°)...\n` +
              `🔬 Processing ${totalProcessed}/${files.length}: ${fileObj.name}`
            );
          } catch (err) {
            console.warn('Could not rotate image, using original:', err);
          }
        }

        const formData = new FormData();
        formData.append('file', fileToProcess);

        let fileSuccess = false;
        let skipDueToDuplicate = false;
        try {
          const response = await fetch(`${API_BASE}/api/extract-tables`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
          }

          const result = await response.json();
          const normalizedTables = normalizeExtractionResponse(result);
          
          if (result.success && normalizedTables.length > 0) {
            const monthKey = monthName;
            const yearKey = selectedYear ? String(selectedYear) : '';
            const duplicateKey = `${fileObj.name}-${monthKey}-${yearKey}`;
            const existingGroup = results.some(r => r.duplicateKey === duplicateKey);
            if (existingGroup) {
              resultDuplicates.push(fileObj.name);
              skipDueToDuplicate = true;
              fileSuccess = false;
              continue;
            }
            normalizedTables.forEach((table, idx) => {
              if (table.dataframe && table.dataframe.length > 0) {
                allResults.push({
                  id: Date.now() + idx + Math.random(),
                  filename: fileObj.name,
                  duplicateKey,
                  fileGroupId: fileObj.id,
                  fileSize: fileObj.size,
                  mode: 'table',
                  tableNumber: idx + 1,
                  totalTables: normalizedTables.length,
                  data: table.dataframe,
                  rowCount: table.row_count,
                  colCount: table.col_count,
                  csvData: table.csv,
                  jsonData: table.json,
                  htmlData: table.html,
                  headers: table.headers || Object.keys(table.dataframe[0] || {}),
                  district: districtName,
                  mandal: mandalName,
                  village: villageName,
                  month: monthName,
                  year: selectedYear,
                  ocrEngine: 'google-vision-api',
                  extractionMethod: 'vision-api-structure',
                  timestamp: new Date().toLocaleString()
                });
              }
            });
            if (!skipDueToDuplicate) {
              successCount++;
              fileSuccess = true;
            }
          } else {
            errorCount++;
            fileSuccess = false;
            console.error('No tables found in response:', result);
            newFailedFiles.push({
              filename: fileObj.name,
              size: fileObj.size,
              error: 'No tables found in the document',
              timestamp: new Date().toLocaleString(),
              month: monthName,
              year: selectedYear
            });
          }
        } catch (err) {
          console.error('Table extraction failed for', fileObj.name, err);
          errorCount++;
          fileSuccess = false;
          
          // Check if it's a connection error
          const isConnectionError = err.message.includes('Failed to fetch') || 
                                   err.message.includes('ERR_CONNECTION_REFUSED') ||
                                   err.name === 'TypeError';
          
          const errorMessage = isConnectionError 
            ? `Backend server is not running. Please start the backend server at ${API_BASE}`
            : (err.message || 'Table extraction failed');
          
          newFailedFiles.push({
            filename: fileObj.name,
            size: fileObj.size,
            error: errorMessage,
            timestamp: new Date().toLocaleString(),
            month: monthName,
            year: selectedYear
          });
          
          // Only show connection error once, not for every file
          if (isConnectionError && totalProcessed === 1) {
            setMessage(
              `❌ Connection Error\n\n` +
              `Cannot connect to backend server at ${API_BASE}\n\n` +
              `Please make sure the backend server is running:\n` +
              `1. Open a terminal in the backend folder\n` +
              `2. Run: python app.py\n` +
              `Or use: npm run dev:all (to start both frontend and backend)`
            );
            setMessageType('error');
          }
        }
      }

      // Update results and failed files
      if (allResults.length > 0) {
        setResults(prev => [...prev, ...allResults]);
      }
      
      if (newFailedFiles.length > 0) {
        setFailedFiles(prev => [...prev, ...newFailedFiles]);
      }

      // Check if all failures were due to connection errors
      const allConnectionErrors = newFailedFiles.length > 0 && 
        newFailedFiles.every(f => f.error.includes('Backend server is not running'));

      if (resultDuplicates.length > 0) {
        setDuplicateResultFiles(resultDuplicates);
        setShowDuplicateResultModal(true);
      } else {
        setDuplicateResultFiles([]);
        setShowDuplicateResultModal(false);
      }

      if (allResults.length > 0) {
        const totalRows = allResults.reduce((sum, r) => sum + (r.rowCount || 0), 0);
        
        const successMsg = `✅ OCR Conversion Completed Successfully!\n\n` +
          `📊 Total Tables Extracted: ${allResults.length}\n` +
          `📋 Total Rows: ${totalRows}\n` +
          `📋 Total Columns: ${allResults[0]?.colCount || 0}\n` +
          `✅ Success: ${successCount} file(s)\n` +
          `${errorCount > 0 ? `❌ Errors: ${errorCount} file(s)\n` : ''}` +
          `📍 Location: ${districtName} → ${mandalName} → ${villageName}`;
        
        setSuccessMessage(successMsg);
        setShowSuccessModal(true);
        setActiveTab('results');
      } else if (allConnectionErrors) {
        // Don't show "no tables extracted" if it's just connection errors
        setActiveTab('results');
      } else if (newFailedFiles.length > 0) {
        setMessage(
          `⚠️ Processing completed but no tables extracted\n` +
          `📁 Processed: ${totalProcessed} file(s)\n` +
          `❌ No valid table data found`
        );
        setMessageType('error');
        setActiveTab('results');
      } else {
        setMessage(
          `⚠️ Processing completed but no tables extracted\n` +
          `📁 Processed: ${totalProcessed} file(s)\n` +
          `❌ No valid table data found`
        );
        setMessageType('error');
      }

      files.forEach(f => {
        if (f.previewUrl) URL.revokeObjectURL(f.previewUrl);
      });
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      console.error('Processing error:', err);
      setMessage(
        `❌ Connection Error\n\n` +
        `Cannot connect to backend server at ${API_BASE}\n\n` +
        `Error: ${err.message}`
      );
      setMessageType('error');
    } finally {
      setProcessing(false);
    }
  };

  const exportAsCSV = (result) => {
    const csvContent = result.csvData || '';
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    
    const filenameBase = result.filename.split('.')[0];
    const monthYearSuffix = result.month && result.year ? `_${result.month}_${result.year}` : '';
    link.download = `${filenameBase}_table_${result.tableNumber}${monthYearSuffix}_export.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const updateResultData = (resultId, updatedFields) => {
    setResults(prev =>
      prev.map(result =>
        result.id === resultId ? { ...result, ...updatedFields } : result
      )
    );
    setSelectedResult(prev =>
      prev && prev.id === resultId ? { ...prev, ...updatedFields } : prev
    );
  };

  const viewResult = (result) => {
    setSelectedResult(result);
  };

  const closeModal = () => {
    setSelectedResult(null);
    setResultZoom(1);
  };

  const formatBytes = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  const getDisplayFileName = (filename = '') => {
    if (!filename) return '';
    return filename.replace(/\.[^/.]+$/, '');
  };

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 overflow-hidden">
      <div className="h-full overflow-y-auto">
        <div className="max-w-7xl mx-auto p-4 lg:p-6">
          {/* Header */}
          <div className="text-center mb-6">
            <h2 className="text-2xl lg:text-4xl font-extrabold mb-2 bg-gradient-to-r from-yellow-300 via-orange-300 to-yellow-300 bg-clip-text text-transparent">
              SOCIETY FOR ELIMINATION OF RURAL POVERTY
            </h2>
            <h3 className="text-lg lg:text-2xl font-semibold text-blue-200">
              Department of Rural Development, Government of Andhra Pradesh
            </h3>
          </div>
          
          {/* Main Header */}
          <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-6 lg:p-8 mb-6 shadow-2xl border border-white/20">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 lg:w-14 lg:h-14 bg-gradient-to-br from-yellow-400 to-orange-600 rounded-xl flex items-center justify-center shadow-lg">
                <Table2 size={28} className="text-white" />
              </div>
              <div className="flex-1">
                <h1 className="text-xl lg:text-2xl font-extrabold text-white mb-2">
                  Digitalizing SHG Data
                </h1>
              </div>
            </div>

            {/* Tab Navigation */}
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => setActiveTab('upload')}
                className={`px-6 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${
                  activeTab === 'upload'
                    ? 'bg-white text-purple-900 shadow-lg'
                    : 'bg-white/20 text-white hover:bg-white/30'
                }`}
              >
                <Upload size={20} />
                 Import Process
              </button>
              <button
                onClick={() => setActiveTab('results')}
                className={`px-6 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${
                  activeTab === 'results'
                    ? 'bg-white text-purple-900 shadow-lg'
                    : 'bg-white/20 text-white hover:bg-white/30'
                }`}
              >
                <Table2 size={20} />
                Converted Results {results.length > 0 && `(${results.length})`}
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`px-6 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${
                  activeTab === 'analytics'
                    ? 'bg-white text-purple-900 shadow-lg'
                    : 'bg-white/20 text-white hover:bg-white/30'
                }`}
              >
                <BarChart3 size={20} />
                Data Analytics
              </button>
              <button
                onClick={() => setActiveTab('shg-financial')}
                className={`px-6 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${
                  activeTab === 'shg-financial'
                    ? 'bg-white text-purple-900 shadow-lg'
                    : 'bg-white/20 text-white hover:bg-white/30'
                }`}
              >
                <TrendingUp size={20} />
                SHG Financial Analytics
              </button>
            </div>
          </div>

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <>
              {/* Location Selection */}
              <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-2xl p-6 border-2 border-red-300 mb-6 shadow-lg">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-orange-500 rounded-xl flex items-center justify-center shadow-md">
                    <MapPin size={28} className="text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl lg:text-2xl font-bold text-red-900">
                      Location Selection <span className="text-red-600">*</span>
                    </h3>
                    <p className="text-sm text-red-700">Select District, Mandal, and Village</p>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="relative">
                    <label className="block text-sm font-bold text-red-900 mb-2">
                      District <span className="text-red-600">*</span>
                    </label>
                    <select
                      value={selectedDistrict}
                      onChange={(e) => setSelectedDistrict(e.target.value)}
                      className="w-full px-4 py-3 text-base font-medium border-2 border-red-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-red-500 bg-white appearance-none cursor-pointer"
                    >
                      <option value="">-- Select District --</option>
                      {districts.map((d) => (
                        <option key={d.id} value={d.id}>{d.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="relative">
                    <label className="block text-sm font-bold text-red-900 mb-2">
                      Mandal <span className="text-red-600">*</span>
                    </label>
                    <select
                      value={selectedMandal}
                      onChange={(e) => setSelectedMandal(e.target.value)}
                      disabled={!mandals.length}
                      className="w-full px-4 py-3 text-base font-medium border-2 border-red-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 bg-white appearance-none cursor-pointer disabled:bg-gray-100 disabled:cursor-not-allowed"
                    >
                      <option value="">-- Select Mandal --</option>
                      {mandals.map((m) => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="relative">
                    <label className="block text-sm font-bold text-red-900 mb-2">
                      Village <span className="text-red-600">*</span>
                    </label>
                    <select
                      value={selectedVillage}
                      onChange={(e) => setSelectedVillage(e.target.value)}
                      disabled={!villages.length}
                      className="w-full px-4 py-3 text-base font-medium border-2 border-red-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 bg-white appearance-none cursor-pointer disabled:bg-gray-100 disabled:cursor-not-allowed"
                    >
                      <option value="">-- Select Village --</option>
                      {villages.map((v) => (
                        <option key={v.id} value={v.id}>{v.name}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Month/Year and Upload Section */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                {/* Month and Year */}
                <div className="lg:col-span-1 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-6 border-2 border-blue-300 shadow-lg">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center shadow-md">
                      <FileText size={28} className="text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg lg:text-xl font-bold text-blue-900">
                        Month & Year <span className="text-blue-600">*</span>
                      </h3>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-bold text-blue-900 mb-2">
                        Month <span className="text-blue-600">*</span>
                      </label>
                      <select
                        value={selectedMonth}
                        onChange={(e) => setSelectedMonth(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white appearance-none cursor-pointer"
                      >
                        <option value="">-- Select Month --</option>
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

                    <div>
                      <label className="block text-sm font-bold text-blue-900 mb-2">
                        Year <span className="text-blue-600">*</span>
                      </label>
                      <select
                        value={selectedYear}
                        onChange={(e) => setSelectedYear(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white appearance-none cursor-pointer"
                      >
                        <option value="">-- Select Year --</option>
                        {Array.from({ length: 10 }, (_, i) => {
                          const year = new Date().getFullYear() - i;
                          return <option key={year} value={year}>{year}</option>;
                        })}
                      </select>
                    </div>

                  </div>
                </div>

                {/* File Upload Section */}
                <div className="lg:col-span-2 bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
                  <h2 className="text-2xl lg:text-3xl font-bold text-gray-800 flex items-center gap-2 mb-6">
                    <Upload size={32} className="text-indigo-600" />
                    Upload Files
                  </h2>

                  {files.length > 0 && (
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-blue-800">{uploadStats.total}</div>
                        <div className="text-xs text-blue-600 font-semibold">Total Files</div>
                      </div>
                      <div className="bg-green-50 border-2 border-green-300 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-green-800">{uploadStats.validated}</div>
                        <div className="text-xs text-green-600 font-semibold">Validated</div>
                      </div>
                      <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-orange-800">{uploadStats.pending}</div>
                        <div className="text-xs text-orange-600 font-semibold">Pending</div>
                      </div>
                    </div>
                  )}

                  <div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    className="border-4 border-dashed border-indigo-300 rounded-2xl p-8 lg:p-12 bg-gradient-to-br from-indigo-50 to-purple-50 hover:from-indigo-100 hover:to-purple-100 transition-all cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div className="text-center">
                      <Upload size={64} className="mx-auto text-indigo-600 mb-4" />
                      <p className="text-xl font-bold text-gray-800 mb-2">
                        Drop files here or click to upload
                      </p>
                      <p className="text-sm text-gray-600">
                        Supports: PNG, JPG, PDF, TIFF (Max 16MB)
                      </p>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept=".png,.jpg,.jpeg,.pdf,.tiff,.tif,.bmp,.webp"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </div>

                  {files.length > 0 && (
                    <div className="mt-6 mb-4 flex justify-end">
                      <button
                        onClick={validateAllFiles}
                        className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-all font-semibold flex items-center gap-2 shadow-md"
                      >
                        <CheckCircle size={20} />
                        Validate All Documents
                      </button>
                    </div>
                  )}

                  {files.length > 0 && (
                    <div className="mt-6 space-y-3">
                      <h3 className="text-lg font-bold text-gray-800">Selected Files ({files.length})</h3>
                      {files.map(fileObj => (
                        <div key={fileObj.id} className="bg-gray-50 border-2 border-gray-300 rounded-lg p-4 flex items-center gap-4">
                          <FileText size={32} className="text-indigo-600 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="font-semibold text-gray-800 truncate">{fileObj.name}</p>
                            <p className="text-sm text-gray-600">{formatBytes(fileObj.size)}</p>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            {fileObj.validated ? (
                              <span className="px-3 py-1 bg-green-500 text-white text-xs rounded-full font-bold flex items-center gap-1">
                                <CheckCircle size={14} />
                                Validated
                              </span>
                            ) : (
                              <button
                                onClick={() => validateFile(fileObj.id)}
                                className="px-3 py-1 bg-yellow-500 hover:bg-yellow-600 text-white text-xs rounded-full font-bold"
                              >
                                Validate
                              </button>
                            )}
                            <button
                              onClick={() => previewFileHandler(fileObj)}
                              className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg"
                            >
                              <Eye size={18} />
                            </button>
                            <button
                              onClick={() => removeFile(fileObj.id)}
                              className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
                            >
                              <X size={18} />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {message && (
                    <div className={`mt-6 p-4 rounded-lg border-2 ${
                      messageType === 'success' ? 'bg-green-50 border-green-300 text-green-800' :
                      messageType === 'error' ? 'bg-red-50 border-red-300 text-red-800' :
                      'bg-blue-50 border-blue-300 text-blue-800'
                    }`}>
                      <p className="whitespace-pre-line font-semibold">{message}</p>
                    </div>
                  )}

                  <div className="mt-6">
                    <button
                      onClick={handleProcess}
                      disabled={processing || files.length === 0}
                      className="w-full px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-bold text-lg transition-all shadow-lg disabled:cursor-not-allowed flex items-center justify-center gap-3"
                    >
                      {processing ? (
                        <>
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                          Processing...
                        </>
                      ) : (
                        <>
                          <Table2 size={24} />
                          Converting into Digital File
                        </>
                      )}
                    </button>
                  </div>

                </div>
              </div>
            </>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && (
            <ConvertedResults
              results={results}
              failedFiles={failedFiles}
              onViewResult={viewResult}
              onExportCSV={exportAsCSV}
              onShowStats={() => {}}
              selectedMonth={selectedMonth}
              selectedYear={selectedYear}
              onUpdateResult={updateResultData}
            />
          )}

          {/* Analytics Tab */}
          {activeTab === 'analytics' && (
            <DataAnalytics
              results={results}
              failedFiles={failedFiles}
              files={files}
              uploadStats={uploadStats}
              districts={districts}
              selectedMonth={selectedMonth}
              selectedYear={selectedYear}
              selectedDistrict={selectedDistrict}
              selectedMandal={selectedMandal}
              selectedVillage={selectedVillage}
            />
          )}

          {/* SHG Financial Analytics Tab */}
          {activeTab === 'shg-financial' && (
            <FinancialAnalytics
              districts={districts}
              selectedMonth={selectedMonth}
              selectedYear={selectedYear}
              selectedDistrict={selectedDistrict}
              selectedMandal={selectedMandal}
              selectedVillage={selectedVillage}
            />
          )}
      </div>
    </div>

      {toasts.length > 0 && (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3">
          {toasts.map(toast => {
            const isSuccess = toast.type === 'success';
            const isWarning = toast.type === 'warning';
            const palette = isSuccess
              ? 'bg-green-50 border-green-300 text-green-900'
              : isWarning
                ? 'bg-yellow-50 border-yellow-300 text-yellow-900'
                : 'bg-blue-50 border-blue-300 text-blue-900';
            return (
              <div
                key={toast.id}
                className={`w-80 border-2 rounded-xl shadow-2xl p-4 backdrop-blur ${palette}`}
              >
                <div className="flex gap-3">
                  <div className="mt-1">
                    {isSuccess ? (
                      <CheckCircle size={22} className="text-green-600" />
                    ) : (
                      <AlertCircle
                        size={22}
                        className={isWarning ? 'text-yellow-600' : 'text-blue-600'}
                      />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-wide">{toast.title}</p>
                    <ul className="mt-1 text-sm leading-relaxed space-y-1">
                      {toast.details.map((line, idx) => (
                        <li key={`${toast.id}-${idx}`} className="break-words">
                          {line}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <button
                    onClick={() => dismissToast(toast.id)}
                    className="text-gray-500 hover:text-gray-700"
                    aria-label="Dismiss notification"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Preview Modal */}
      {previewFile && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b-2 border-gray-300">
              <h3 className="text-xl font-bold text-gray-800">{previewFile.name}</h3>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => rotateImage('left')}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded transition-all"
                  >
                    <RotateCcw size={20} />
                  </button>
                  <button
                    onClick={() => rotateImage('right')}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded transition-all"
                  >
                    <RotateCw size={20} />
                  </button>
                </div>
                <button
                  onClick={closePreview}
                  className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
                >
                  <X size={24} />
                </button>
              </div>
            </div>
            <div className="p-4 overflow-auto max-h-[calc(90vh-80px)] flex items-center justify-center">
              {previewFile.previewUrl && (
                <img 
                  src={previewFile.previewUrl} 
                  alt={previewFile.name}
                  className="max-w-full h-auto mx-auto"
                  style={{ 
                    transform: `rotate(${previewFile.rotation || 0}deg) scale(${previewZoom})`,
                    maxHeight: 'calc(90vh - 120px)'
                  }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Result Detail Modal */}
      {selectedResult && (
        <div className="fixed inset-0 bg-black/80 z-50 flex flex-col">
          <div className="bg-white shadow-2xl flex flex-col h-full w-full">
            <div className="flex items-center justify-between p-6 border-b-2 border-gray-300">
              <div>
                <h3 className="text-2xl font-bold text-gray-800 truncate">{getDisplayFileName(selectedResult.filename)}</h3>
                <p className="text-sm text-gray-600 mt-2 flex items-center gap-2">
                  📍 {selectedResult.district} → {selectedResult.mandal} → {selectedResult.village}
                  {selectedResult.month && selectedResult.year && (
                    <span> | 📅 {selectedResult.month} {selectedResult.year}</span>
                  )}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => setResultZoom(prev => Math.max(prev - 0.25, 0.5))}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded"
                    disabled={resultZoom <= 0.5}
                  >
                    <ZoomOut size={20} />
                  </button>
                  <button
                    onClick={() => setResultZoom(1)}
                    className="px-3 py-2 bg-white hover:bg-gray-200 text-gray-700 rounded text-sm font-semibold"
                  >
                    {Math.round(resultZoom * 100)}%
                  </button>
                  <button
                    onClick={() => setResultZoom(prev => Math.min(prev + 0.25, 3))}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded"
                    disabled={resultZoom >= 3}
                  >
                    <ZoomIn size={20} />
                  </button>
                </div>
                <button
                  onClick={closeModal}
                  className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
                >
                  <X size={24} />
                </button>
              </div>
            </div>
            <div className="p-6 flex-1 overflow-auto result-modal-content" style={{ minHeight: 0, overflowX: 'auto', overflowY: 'auto' }}>
              <style>{`
                .result-modal-content .tables-wrapper {
                  max-height: none !important;
                  overflow-y: visible !important;
                  overflow-x: visible !important;
                  height: auto !important;
                  width: 100% !important;
                }
                .result-modal-content .table-section {
                  margin-bottom: 30px;
                }
                .result-modal-content .shg-table {
                  width: 100% !important;
                }
              `}</style>
              {selectedResult.htmlData ? (
                <div 
                  className="w-full"
                  style={{
                    transformOrigin: 'top left',
                    transform: `scale(${resultZoom})`,
                    minWidth: `${100 / resultZoom}%`
                  }}
                >
                  <div
                    style={{
                      width: '100%'
                    }}
                    dangerouslySetInnerHTML={{ __html: selectedResult.htmlData }}
                  />
                </div>
              ) : (
                <div 
                  className="w-full"
                  style={{
                    transformOrigin: 'top left',
                    transform: `scale(${resultZoom})`,
                    minWidth: `${100 / resultZoom}%`
                  }}
                >
                  <table className="w-full border-collapse border-2 border-gray-300">
                    <thead>
                      <tr className="bg-indigo-700 text-white">
                        {selectedResult.headers.map((header, idx) => (
                          <th key={idx} className="border-2 border-gray-300 px-2 py-2 text-center font-bold">
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {selectedResult.data.map((row, rowIdx) => (
                        <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                          {selectedResult.headers.map((header, cellIdx) => (
                            <td key={cellIdx} className="border-2 border-gray-300 px-2 py-2 text-sm">
                              {row[header] || ''}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Duplicate Modal */}
      {showDuplicateModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                <AlertCircle size={32} className="text-orange-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-800">Duplicate Files!</h3>
            </div>
            <p className="text-gray-700 mb-4">The following files are already uploaded:</p>
            <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-4 mb-6 max-h-48 overflow-y-auto">
              {duplicateFiles.map((name, idx) => (
                <p key={idx} className="text-sm text-orange-800 font-semibold py-1">• {name}</p>
              ))}
            </div>
            <button
              onClick={() => {
                setShowDuplicateModal(false);
                setDuplicateFiles([]);
              }}
              className="w-full px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg font-bold"
            >
              OK, Got it
            </button>
          </div>
        </div>
      )}

      {/* Success Modal */}
      {showSuccessModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle size={40} className="text-green-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-800">Success!</h3>
                <p className="text-sm text-green-600 font-semibold">OCR conversion completed</p>
              </div>
            </div>
            <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4 mb-6">
              <p className="text-gray-800 whitespace-pre-line font-semibold text-sm">{successMessage}</p>
            </div>
            <button
              onClick={() => setShowSuccessModal(false)}
              className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-bold"
            >
              View Results
            </button>
          </div>
        </div>
      )}

      {showDuplicateResultModal && duplicateResultFiles.length > 0 && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                <AlertCircle size={32} className="text-yellow-600" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800">Duplicate Files Detected</h3>
                <p className="text-sm text-yellow-700">These files were already uploaded this month:</p>
              </div>
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 max-h-48 overflow-y-auto mb-6">
              <ul className="list-disc list-inside text-sm text-yellow-900 space-y-1">
                {duplicateResultFiles.map((name, idx) => (
                  <li key={`${name}-${idx}`}>{name}</li>
                ))}
              </ul>
            </div>
            <button
              onClick={() => setShowDuplicateResultModal(false)}
              className="w-full px-4 py-3 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg font-bold"
            >
              OK, Got it
            </button>
          </div>
        </div>
      )}

      {showDuplicateUploadModal && duplicateUploadFiles.length > 0 && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                <AlertCircle size={32} className="text-yellow-600" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800">Duplicate Upload</h3>
                <p className="text-sm text-yellow-700">These files already exist for the selected month:</p>
              </div>
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 max-h-48 overflow-y-auto mb-6">
              <ul className="list-disc list-inside text-sm text-yellow-900 space-y-1">
                {duplicateUploadFiles.map((name, idx) => (
                  <li key={`${name}-${idx}`}>{name}</li>
                ))}
              </ul>
            </div>
            <button
              onClick={() => {
                setShowDuplicateUploadModal(false);
                setDuplicateUploadFiles([]);
              }}
              className="w-full px-4 py-3 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg font-bold"
            >
              OK, Got it
            </button>
          </div>
        </div>
      )}
    </div>
  );
}