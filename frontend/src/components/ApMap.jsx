import React, { useMemo, useRef, useState, useEffect } from 'react';
import apDistricts from '../data/ap-districts';
import { MapPin } from 'lucide-react';

const normalizeDistrictName = (name = '') => name.toLowerCase().replace(/[^a-z0-9]/g, '');

// Precompute canonical names for each AP district
const canonicalDistrictNameToId = apDistricts.reduce((acc, district) => {
  acc[normalizeDistrictName(district.district)] = district.id;
  return acc;
}, {});

// Allow alternate spellings commonly seen in Excel/map labels
const districtNameAliases = {
  sripottisriramulunellore: 'spsr_nellore',
  sripottisriramulunelloredistrict: 'spsr_nellore',
  nellore: 'spsr_nellore',
  sripottisriramulu: 'spsr_nellore',
  ysrkadapa: 'kadapa',
  kadapa: 'kadapa',
  kadapadistrict: 'kadapa',
  vishakhapatnam: 'visakhapatnam',
  visakhapatnam: 'visakhapatnam',
  visakapatanam: 'visakhapatnam',
  anantapuram: 'anantapur',
  ananthapur: 'anantapur',
  'sri satyasaidistrict': 'anantapur',
  prakasam: 'prakasam',
  guntur: 'guntur',
  westgodavari: 'west_godavari',
  eastgodavari: 'east_godavari',
  vizianagaram: 'vizianagaram',
  srikakulam: 'srikakulam',
  krishna: 'krishna',
  kurnool: 'kurnool',
  chittoor: 'chittoor',
};

const resolveDistrictId = (name = '') => {
  const normalized = normalizeDistrictName(name);
  return canonicalDistrictNameToId[normalized] || districtNameAliases[normalized] || null;
};

// Mapping from SVG encoded path IDs to district IDs (resolved from the provided SVG)
const districtPathMapping = {
  '_x31_58-3': 'srikakulam',
  '_x31_59-6': 'vizianagaram',
  '_x33_56-0': 'visakhapatnam',
  'path1056': 'east_godavari',
  'path1034': 'west_godavari',
  '_x31_80-3': 'krishna',
  '_x31_82-5': 'guntur',
  '_x31_83-6': 'prakasam',
  '_x31_89-2': 'spsr_nellore',
  '_x31_87-9': 'kadapa',
  '_x31_84-2': 'kurnool',
  '_x31_88-1': 'anantapur',
  '_x31_90-7': 'chittoor',
};

const IMAGE_CANDIDATES = [
  '/OCR/andhra-pradesh-map-image.png',
  '/OCR/andhra-pradesh-map-image.jpg',
  '/OCR/andhra-pradesh-map-image.jpeg',
];

const numberFormatter = new Intl.NumberFormat('en-IN');
const currencyFormatter = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
});
const districtTotalsConfig = [
  { label: 'Total Saving Balance', key: 'Total Savings Balance', format: 'currency', accent: 'from-indigo-50 to-indigo-100 border-indigo-200 text-indigo-900' },
  { label: 'Total - SHG Loan Balance', key: 'Total - SHG Loan Balance', format: 'currency', accent: 'from-blue-50 to-blue-100 border-blue-200 text-blue-900' },
  { label: 'Total - Bank Loan Balance', key: 'Total - Bank Loan Balance', format: 'currency', accent: 'from-sky-50 to-sky-100 border-sky-200 text-sky-900' },
  { label: 'Total - Srinidi Micro Loan Balance', key: 'Total - Srinidi Micro Loan Balance', format: 'currency', accent: 'from-emerald-50 to-emerald-100 border-emerald-200 text-emerald-900' },
  { label: 'Total - Srinidi Tenny Loan Balance', key: 'Total - Srinidi Tenny Loan Balance', format: 'currency', accent: 'from-lime-50 to-lime-100 border-lime-200 text-lime-900' },
  { label: 'Total - Unnati SCSP Loan Balance', key: 'Total - Unnati SCSP Loan Balance', format: 'currency', accent: 'from-amber-50 to-amber-100 border-amber-200 text-amber-900' },
  { label: 'Total - Unnati TSP Loan Balance', key: 'Total - Unnati TSP Loan Balance', format: 'currency', accent: 'from-orange-50 to-orange-100 border-orange-200 text-orange-900' },
  { label: 'Total - CIF Loan Balance', key: 'Total - CIF Loan Balance', format: 'currency', accent: 'from-rose-50 to-rose-100 border-rose-200 text-rose-900' },
  { label: 'Total - VO Loan Balance', key: 'Total - VO Loan Balance', format: 'currency', accent: 'from-purple-50 to-purple-100 border-purple-200 text-purple-900' },
  { label: 'New Loan Type', key: 'New Loan Type', format: 'number', accent: 'from-fuchsia-50 to-fuchsia-100 border-fuchsia-200 text-fuchsia-900' },
  { label: 'New Total', key: 'New Total', format: 'currency', accent: 'from-slate-50 to-slate-100 border-slate-200 text-slate-900' },
];
const formatNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numberFormatter.format(numeric) : '—';
};
const formatCurrency = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? currencyFormatter.format(numeric) : '—';
};
const formatValue = (value, type = 'number') => {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return '—';
  }
  return type === 'currency' ? formatCurrency(value) : formatNumber(value);
};

const ApMap = ({
  selectedDistrictId: controlledSelectedDistrictId,
  onDistrictSelect,
  districtSummaries = {},
  isAnalyticsLoading = false,
}) => {
  const [internalSelectedDistrictId, setInternalSelectedDistrictId] = useState(null);
  const [svgContent, setSvgContent] = useState(null);
  const [svgError, setSvgError] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [useImageMap, setUseImageMap] = useState(false); // Prefer interactive SVG by default
  const [imageCheckKey, setImageCheckKey] = useState(0);
  const [processedSvg, setProcessedSvg] = useState(null);
  const mapWrapperRef = useRef(null);
  const imageRef = useRef(null);
  const svgContainerRef = useRef(null);
  const svgMarkup = useMemo(() => {
    if (!processedSvg) return null;
    return { __html: processedSvg };
  }, [processedSvg]);

  const activeDistrictId = controlledSelectedDistrictId ?? internalSelectedDistrictId;

  const districtSummariesById = useMemo(() => {
    if (!districtSummaries) return {};
    const entries = Array.isArray(districtSummaries)
      ? districtSummaries
      : Object.values(districtSummaries);

    return entries.reduce((acc, summary) => {
      if (!summary?.district) return acc;
      const districtId = resolveDistrictId(summary.district);
      if (!districtId) return acc;

      const canonicalName =
        apDistricts.find((district) => district.id === districtId)?.district ||
        summary.district;

      acc[districtId] = {
        ...summary,
        district: canonicalName,
      };
      return acc;
    }, {});
  }, [districtSummaries]);

  const selectedDistrict = useMemo(() => {
    return apDistricts.find((district) => district.id === activeDistrictId) || null;
  }, [activeDistrictId]);

  const analyticsSummary = districtSummariesById[activeDistrictId];
  const aggregatedSummary = useMemo(() => {
    const summaries = Object.values(districtSummariesById);
    if (!summaries.length) return null;

    const combinedTotals = districtTotalsConfig.reduce((acc, { key }) => {
      acc[key] = 0;
      return acc;
    }, {});

    summaries.forEach((summary) => {
      districtTotalsConfig.forEach(({ key }) => {
        const value = Number(summary.column_totals?.[key]) || 0;
        combinedTotals[key] += value;
      });
    });

    return {
      district: 'Andhra Pradesh',
      column_totals: combinedTotals,
    };
  }, [districtSummariesById]);

  const displaySummary = (activeDistrictId && analyticsSummary) ? analyticsSummary : aggregatedSummary;
  const displayName = activeDistrictId && selectedDistrict
    ? selectedDistrict.district
    : displaySummary?.district || 'Select a district';

  const handleMapClick = React.useCallback((event) => {
    // Handle image map clicks (coordinate-based selection)
    if (useImageMap && imageRef.current && imageSrc) {
      const img = imageRef.current;
      const rect = img.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      // Calculate relative position (0-1)
      const relX = x / rect.width;
      const relY = y / rect.height;
      
      // For now, allow manual district selection via dropdown or use coordinate mapping
      // This is a placeholder - you can add coordinate-based district detection here
      console.log('Image map clicked at:', relX, relY);
      return;
    }
    
    // Handle SVG path clicks
    const targetPath = event.target.closest('path');
    if (!targetPath || !targetPath.classList.contains('district')) return;
    
    // Try to get district ID from path
    const pathId = targetPath.dataset.districtId || targetPath.id || targetPath.getAttribute('data-district-id');
    const updateSelection = (districtId) => {
      setInternalSelectedDistrictId(districtId);
      if (onDistrictSelect) {
        const districtName =
          apDistricts.find((district) => district.id === districtId)?.district ||
          districtSummariesById[districtId]?.district ||
          districtId;
        onDistrictSelect(districtId, districtName);
      }
    };

    const match = apDistricts.find((district) => district.id === pathId);
    
    if (match) {
      updateSelection(match.id);
    } else {
      // Try to find by mapping
      const originalId = targetPath.getAttribute('data-original-id');
      const mappedDistrictId = districtPathMapping[originalId] || districtPathMapping[pathId];
      
      if (mappedDistrictId) {
        const mappedMatch = apDistricts.find((district) => district.id === mappedDistrictId);
        if (mappedMatch) {
          updateSelection(mappedMatch.id);
          return;
        }
      }
      
      // Log for debugging - helps identify which paths need mapping
      console.log('Clicked path - ID:', pathId, 'Original ID:', originalId, 'Path data length:', targetPath.getAttribute('d')?.length);
    }
  }, [useImageMap, imageSrc, onDistrictSelect, districtSummariesById]);

  // Detect which image file exists (png/jpg/jpeg) and prefer using it
  useEffect(() => {
    let isMounted = true;

    const tryLoadImages = async () => {
      for (const candidate of IMAGE_CANDIDATES) {
        try {
          await new Promise((resolve, reject) => {
            const testImage = new Image();
            testImage.onload = () => resolve();
            testImage.onerror = reject;
            testImage.src = candidate + '?t=' + Date.now(); // cache bust
          });

          if (isMounted) {
            setImageSrc(candidate);
            setUseImageMap(true);
          }
          return;
        } catch (error) {
          // try next candidate
        }
      }

      if (isMounted) {
        setImageSrc(null);
        setUseImageMap(false);
      }
    };

    tryLoadImages();

    return () => {
      isMounted = false;
    };
  }, [imageCheckKey]);

  const retryImageMapDetection = () => {
    setImageSrc(null);
    setUseImageMap(true);
    setImageCheckKey((key) => key + 1);
  };

  // Load SVG content
  useEffect(() => {
    // Try multiple paths - first without /OCR/, then with /OCR/
    const svgPaths = [
      '/andhra-pradesh-map.svg',
      '/OCR/andhra-pradesh-map.svg'
    ];
    
    let currentPathIndex = 0;
    
    const tryFetch = (path) => {
      return fetch(path)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to fetch SVG: ${response.status} ${response.statusText}`);
          }
          // Check content type
          const contentType = response.headers.get('content-type');
          if (contentType && !contentType.includes('svg') && !contentType.includes('xml') && !contentType.includes('text')) {
            throw new Error(`Unexpected content type: ${contentType}. Expected SVG.`);
          }
          return response.text();
        })
        .then((text) => {
          if (!text || !text.trim()) {
            throw new Error('Empty SVG content received.');
          }
          
          const trimmed = text.trim();
          
          // Check if it's HTML (404 page, etc.)
          if (trimmed.startsWith('<!DOCTYPE') || trimmed.startsWith('<html') || (trimmed.startsWith('<!') && !trimmed.startsWith('<?xml'))) {
            throw new Error('Received HTML instead of SVG. The SVG file may not exist.');
          }
          
          // Check if it's actually SVG
          if (!trimmed.startsWith('<svg') && !trimmed.startsWith('<?xml')) {
            throw new Error('Invalid SVG content. File does not appear to be a valid SVG.');
          }
          
          setSvgContent(text);
          setSvgError(null);
        });
    };
    
    const attemptFetch = () => {
      if (currentPathIndex >= svgPaths.length) {
        setSvgError('Failed to load SVG from all attempted paths. Please ensure andhra-pradesh-map.svg exists in the public folder.');
        setUseImageMap(true);
        return;
      }
      
      tryFetch(svgPaths[currentPathIndex])
        .catch((error) => {
          console.warn(`Failed to load SVG from ${svgPaths[currentPathIndex]}:`, error.message);
          currentPathIndex++;
          attemptFetch();
        });
    };
    
    attemptFetch();
  }, []);

  // Process SVG and map district paths after load
  useEffect(() => {
    if (!svgContent || useImageMap) {
      setProcessedSvg(null);
      return;
    }

    try {
      // Clean the SVG content - remove BOM if present
      let cleanSvgContent = svgContent.trim();
      if (cleanSvgContent.charCodeAt(0) === 0xFEFF) {
        cleanSvgContent = cleanSvgContent.slice(1);
      }
      
      // Ensure it starts with SVG
      if (!cleanSvgContent.startsWith('<svg') && !cleanSvgContent.startsWith('<?xml')) {
        throw new Error('SVG content does not start with <svg> or <?xml>');
      }
      
      // Parse and modify SVG
      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(cleanSvgContent, 'image/svg+xml');
      
      // Check for parsing errors
      const parserError = svgDoc.querySelector('parsererror');
      if (parserError) {
        const errorText = parserError.textContent || 'Unknown parsing error';
        console.error('SVG parsing error:', errorText);
        // Log the first few characters to help debug
        console.error('First 200 chars of SVG:', cleanSvgContent.substring(0, 200));
        setSvgError('SVG parsing error: ' + errorText);
        setProcessedSvg(null);
        // Fall back to image map
        setUseImageMap(true);
        return;
      }
      
      const svgElement = svgDoc.documentElement;

      // Find all paths in the SVG
      const paths = svgElement.querySelectorAll('path[id]');
      
      // Map encoded IDs to district IDs (this mapping may need adjustment based on actual SVG structure)
      // The SVG has paths with IDs like "_x31_58-3", "_x31_59-6", etc.
      // We'll need to identify which path corresponds to which district
      // For now, we'll add a data attribute to help identify districts
      paths.forEach((path) => {
        // Add district class to all main district paths (paths with complex d attributes)
        const d = path.getAttribute('d') || '';
        if (d.length > 100) { // Main district paths typically have long path data
          path.setAttribute('class', 'district');
          // Store original ID for reference
          path.setAttribute('data-original-id', path.id);
          const mappedDistrict = districtPathMapping[path.id];
          if (mappedDistrict) {
            path.setAttribute('data-district-id', mappedDistrict);
            path.setAttribute('id', mappedDistrict);
          } else {
            path.setAttribute('data-district-id', path.id);
          }
        }
      });

      // Update viewBox and class
      const viewBox = svgElement.getAttribute('viewBox') || '0 0 1238.0786 1041.3174';
      svgElement.setAttribute('viewBox', viewBox);
      svgElement.setAttribute('class', 'w-full h-auto max-w-4xl');

      // Inject styles
      const style = svgDoc.createElementNS('http://www.w3.org/2000/svg', 'style');
      style.textContent = `
            .district {
              fill: #ffffff;
              stroke: #000000;
              stroke-width: 2.5;
              cursor: pointer;
              transition: all 0.3s ease;
            }
            .district:hover {
              fill: #e0e0e0;
              stroke: #000000;
              stroke-width: 3.5;
            }
            .district.active {
              fill: #FFA500;
              stroke: #000000;
              stroke-width: 4;
            }
      `;
      svgElement.insertBefore(style, svgElement.firstChild);

      // Convert to string for React to render
      const serializer = new XMLSerializer();
      const svgString = serializer.serializeToString(svgElement);
      setProcessedSvg(svgString);
      setSvgError(null);
    } catch (error) {
      console.error('Error processing SVG:', error);
      setSvgError('Error processing SVG: ' + error.message);
      setProcessedSvg(null);
    }
  }, [svgContent, useImageMap]);

  // Add click event listeners to SVG paths after render
  useEffect(() => {
    if (!processedSvg || !svgContainerRef.current) return;

    const container = svgContainerRef.current;
    const paths = container.querySelectorAll('path.district');
    
    const clickHandler = (e) => {
      e.stopPropagation();
      handleMapClick(e);
    };
    
    paths.forEach((path) => {
      path.addEventListener('click', clickHandler);
    });

    // Cleanup function
    return () => {
      paths.forEach((path) => {
        path.removeEventListener('click', clickHandler);
      });
    };
  }, [processedSvg, handleMapClick]);

  useEffect(() => {
    if (!svgContainerRef.current || !processedSvg) return;
    const paths = svgContainerRef.current.querySelectorAll('path.district');
    paths.forEach((path) => {
      const pathId = path.getAttribute('data-district-id') || path.id;
      if (pathId === activeDistrictId) {
        path.classList.add('active');
      } else {
        path.classList.remove('active');
      }
    });
  }, [activeDistrictId, processedSvg]);

  const hasAnalytics = Boolean(displaySummary);
  const columnTotals = displaySummary?.column_totals || {};

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Map Section */}
      <div
        ref={mapWrapperRef}
        className="lg:col-span-2 bg-gradient-to-br from-gray-50 to-gray-100 rounded-3xl shadow-2xl p-6 flex items-center justify-center border-2 border-gray-200"
        onClick={handleMapClick}
      >
        <div 
          className="svg-container w-full h-auto max-w-4xl flex items-center justify-center"
          role="img"
          aria-label="Andhra Pradesh District Map"
        >
          {useImageMap && imageSrc ? (
            // Display image map
            <img
              ref={imageRef}
              src={imageSrc}
              alt="Andhra Pradesh District Map"
              className="w-full h-auto max-w-4xl cursor-pointer"
              onError={(e) => {
                console.error('Image map not found, falling back to SVG');
                setUseImageMap(false);
                setImageSrc(null);
              }}
              onClick={handleMapClick}
              style={{ imageRendering: 'high-quality' }}
            />
          ) : useImageMap ? (
            <div className="text-gray-500">Looking for image map...</div>
          ) : svgError ? (
            <div className="text-red-600 p-4 text-center max-w-md">
              <p className="font-semibold mb-2">Map Loading Error</p>
              <p className="text-sm">{svgError}</p>
              <div className="flex gap-2 mt-4 justify-center">
                <button 
                  onClick={() => {
                    setSvgError(null);
                    setUseImageMap(false);
                    // Try both paths
                    const tryPaths = ['/andhra-pradesh-map.svg', '/OCR/andhra-pradesh-map.svg'];
                    let pathIndex = 0;
                    const attemptFetch = () => {
                      if (pathIndex >= tryPaths.length) {
                        setSvgError('SVG not found. Falling back to image map.');
                        setUseImageMap(true);
                        return;
                      }
                      fetch(tryPaths[pathIndex])
                        .then((response) => {
                          if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`);
                          return response.text();
                        })
                        .then((text) => {
                          if (!text || !text.trim()) {
                            throw new Error('Invalid SVG content');
                          }
                          const trimmed = text.trim();
                          if (trimmed.startsWith('<!DOCTYPE') || trimmed.startsWith('<html')) {
                            throw new Error('Received HTML instead of SVG');
                          }
                          if (!trimmed.startsWith('<svg') && !trimmed.startsWith('<?xml')) {
                            throw new Error('Invalid SVG format');
                          }
                          setSvgContent(text);
                          setSvgError(null);
                        })
                        .catch((error) => {
                          console.warn(`Failed from ${tryPaths[pathIndex]}:`, error.message);
                          pathIndex++;
                          attemptFetch();
                        });
                    };
                    attemptFetch();
                  }}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
                >
                  Retry SVG
                </button>
                <button 
                  onClick={retryImageMapDetection}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                >
                  Try Image Map
                </button>
              </div>
            </div>
          ) : svgMarkup ? (
            <div
              ref={svgContainerRef}
              dangerouslySetInnerHTML={svgMarkup}
              className="w-full h-auto max-w-4xl"
            />
          ) : !svgContent ? (
            <div className="text-gray-500">Loading map...</div>
          ) : null}
        </div>
      </div>

      {/* District Details Sidebar */}
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-3xl shadow-2xl p-6 border-2 border-gray-100">
        <div className="flex items-center gap-3 mb-6">
          <div className="bg-indigo-600 p-3 rounded-2xl">
            <MapPin size={28} className="text-white" />
          </div>
          <h3 className="text-2xl font-bold text-gray-800">District Snapshot</h3>
        </div>
        
        {hasAnalytics ? (
          <div className="space-y-4">
            {/* District Name */}
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-5 shadow-lg">
              <p className="text-sm text-indigo-100 font-semibold uppercase tracking-wider mb-1">Selected District</p>
              <p className="text-3xl font-extrabold text-white">{displayName}</p>
            </div>

            {isAnalyticsLoading && !activeDistrictId && (
              <div className="flex items-center gap-3 bg-indigo-50 border border-indigo-100 text-indigo-700 rounded-2xl px-4 py-3">
                <div className="h-3 w-3 bg-indigo-600 rounded-full animate-ping" />
                <p className="text-sm font-medium">Aggregating statewide totals from the latest Excel upload...</p>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {districtTotalsConfig.map(({ label, key, format, accent }) => (
                <div
                  key={key}
                  className={`bg-gradient-to-br ${accent} rounded-2xl p-4 border shadow-md`}
                >
                  <p className="text-xs uppercase tracking-wide font-bold text-gray-600">{label}</p>
                  <p className="text-base font-extrabold text-gray-900 mt-1 whitespace-nowrap overflow-hidden text-ellipsis">
                    {formatValue(columnTotals[key], format)}
                  </p>
                </div>
              ))}
            </div>

            {/* Info Message */}
            <div className="bg-gray-100 rounded-xl p-4 border border-gray-200">
              <p className="text-sm text-gray-600 text-center">
                💡 Click on any district on the map to view its detailed statistics
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <MapPin size={48} className="mx-auto text-gray-300 mb-4" />
            <p className="text-gray-500 text-lg">Select a district on the map to view details.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ApMap;

