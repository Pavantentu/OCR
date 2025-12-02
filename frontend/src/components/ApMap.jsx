import React, { useEffect, useMemo, useRef, useState } from 'react';
import apDistricts from '../data/ap-districts';
import { MapPin } from 'lucide-react';

const normalizeDistrictName = (name = '') =>
  name.toLowerCase().replace(/[^a-z0-9]/g, '');

// Precompute canonical names for each AP district
const canonicalDistrictNameToId = apDistricts.reduce((acc, district) => {
  acc[normalizeDistrictName(district.district)] = district.id;
  return acc;
}, {});

// Allow alternate spellings commonly seen in Excel/map labels
const districtNameAliases = {
  sripottisriramulunellore: 'spsr_nellore',
  sripottisriramalunellore: 'spsr_nellore',
  nellore: 'spsr_nellore',
  ysrkadapa: 'ysr_kadapa',
  kadapa: 'ysr_kadapa',
  vishakhapatnam: 'visakhapatnam',
  visakhapatnam: 'visakhapatnam',
  visakapatanam: 'visakhapatnam',
  anantapuram: 'anantapur',
  ananthapur: 'anantapur',
  'sri satyasaidistrict': 'sri_sathya_sai',
  srisathyasai: 'sri_sathya_sai',
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

const numberFormatter = new Intl.NumberFormat('en-IN');
const currencyFormatter = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
});

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

// Totals for Balances view
const balanceTotalsConfig = [
  {
    label: 'Total Savings Balance',
    key: 'Total Savings Balance',
    format: 'currency',
    accent:
      'from-indigo-50 to-indigo-100 border-indigo-200 text-indigo-900',
  },
  {
    label: 'Total - SHG Loan Balance',
    key: 'Total - SHG Loan Balance',
    format: 'currency',
    accent: 'from-blue-50 to-blue-100 border-blue-200 text-blue-900',
  },
  {
    label: 'Total - Bank Loan Balance',
    key: 'Total - Bank Loan Balance',
    format: 'currency',
    accent: 'from-sky-50 to-sky-100 border-sky-200 text-sky-900',
  },
  {
    label: 'Total - Streenidhi Micro Loan Balance',
    key: 'Total - Streenidhi Micro Loan Balance',
    format: 'currency',
    accent:
      'from-emerald-50 to-emerald-100 border-emerald-200 text-emerald-900',
  },
  {
    label: 'Total - Streenidhi Tenny Loan Balance',
    key: 'Total - Streenidhi Tenny Loan Balance',
    format: 'currency',
    accent: 'from-lime-50 to-lime-100 border-lime-200 text-lime-900',
  },
  {
    label: 'Total - Unnathi SCSP Loan Balance',
    key: 'Total - Unnathi SCSP Loan Balance',
    format: 'currency',
    accent: 'from-amber-50 to-amber-100 border-amber-200 text-amber-900',
  },
  {
    label: 'Total - Unnathi TSP Loan Balance',
    key: 'Total - Unnathi TSP Loan Balance',
    format: 'currency',
    accent:
      'from-orange-50 to-orange-100 border-orange-200 text-orange-900',
  },
  {
    label: 'Total - CIF Loan Balance',
    key: 'Total - CIF Loan Balance',
    format: 'currency',
    accent: 'from-rose-50 to-rose-100 border-rose-200 text-rose-900',
  },
  {
    label: 'Total - VO Loan Balance',
    key: 'Total - VO Loan Balance',
    format: 'currency',
    accent:
      'from-purple-50 to-purple-100 border-purple-200 text-purple-900',
  },
  {
    label: 'New Loans Disbursed (Total)',
    key: 'New Total',
    format: 'currency',
    accent:
      'from-slate-50 to-slate-100 border-slate-200 text-slate-900',
  },
];

// Totals for Current Month view
const monthTotalsConfig = [
  {
    label: 'New Loans Disbursed (Total)',
    key: 'New Total',
    format: 'currency',
    accent:
      'from-slate-50 to-slate-100 border-slate-200 text-slate-900',
  },
  {
    label: 'This Month Savings',
    key: 'This Month Savings',
    format: 'currency',
    accent:
      'from-emerald-50 to-emerald-100 border-emerald-200 text-emerald-900',
  },
  {
    label: 'SHG Loan Paid',
    key: 'This Month SHG Paid Loan',
    format: 'currency',
    accent: 'from-indigo-50 to-indigo-100 border-indigo-200 text-indigo-900',
  },
  {
    label: 'SHG Bank Loan Paid',
    key: 'This Month Bank Loan Paid',
    format: 'currency',
    accent: 'from-blue-50 to-blue-100 border-blue-200 text-blue-900',
  },
  {
    label: 'Streenidhi Micro Loan Paid',
    key: 'This Month Streenidhi Micro Loan Paid',
    format: 'currency',
    accent: 'from-sky-50 to-sky-100 border-sky-200 text-sky-900',
  },
  {
    label: 'Streenidhi Tenny Loan Paid',
    key: 'This Month Streenidhi Tenny Loan Paid',
    format: 'currency',
    accent: 'from-lime-50 to-lime-100 border-lime-200 text-lime-900',
  },
  {
    label: 'Unnathi SCSP Loan Paid',
    key: 'This Month Unnathi SCSP Loan Paid',
    format: 'currency',
    accent: 'from-amber-50 to-amber-100 border-amber-200 text-amber-900',
  },
  {
    label: 'Unnathi TSP Loan Paid',
    key: 'This Month Unnathi TSP Loan Paid',
    format: 'currency',
    accent:
      'from-orange-50 to-orange-100 border-orange-200 text-orange-900',
  },
  {
    label: 'CIF Loan Paid',
    key: 'This Month CIF Loan Paid',
    format: 'currency',
    accent: 'from-rose-50 to-rose-100 border-rose-200 text-rose-900',
  },
  {
    label: 'VO Loan Paid',
    key: 'This Month VO Loan Paid',
    format: 'currency',
    accent:
      'from-purple-50 to-purple-100 border-purple-200 text-purple-900',
  },
];

const others = [
  {
    label: 'Penalty (Current Month)',
    key: 'Penalties',
    format: 'currency',
    accent: 'from-red-50 to-red-100 border-red-200 text-red-900',
  },
  {
    label: 'Membership Entry Fee (Current Month)',
    key: 'Entry Membership Fee',
    format: 'currency',
    accent: 'from-yellow-50 to-yellow-100 border-yellow-200 text-yellow-900',
  },
  {
    label: 'Savings Returned (Current Month)',
    key: 'Savings Returned',
    format: 'currency',
    accent: 'from-teal-50 to-teal-100 border-teal-200 text-teal-900',
  },
];

const ApMap = ({
  selectedDistrictId: controlledSelectedDistrictId,
  onDistrictSelect,
  districtSummaries = {},
  isAnalyticsLoading = false,
}) => {
  const [internalSelectedDistrictId, setInternalSelectedDistrictId] =
    useState(null);
  const [districtViewMode, setDistrictViewMode] = useState('currentMonth'); // 'currentMonth' | 'balances'
  const [svgContent, setSvgContent] = useState(null);
  const [processedSvg, setProcessedSvg] = useState(null);
  const [svgError, setSvgError] = useState(null);

  const svgContainerRef = useRef(null);

  const svgMarkup = useMemo(() => {
    if (!processedSvg) return null;
    return { __html: processedSvg };
  }, [processedSvg]);

  // Keep internal selection in sync with external props and reset view to Current Month
  useEffect(() => {
    if (controlledSelectedDistrictId) {
      setInternalSelectedDistrictId(controlledSelectedDistrictId);
    }
    setDistrictViewMode('currentMonth');
  }, [controlledSelectedDistrictId]);

  const activeDistrictId =
    controlledSelectedDistrictId ?? internalSelectedDistrictId;

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
    return (
      apDistricts.find((district) => district.id === activeDistrictId) || null
    );
  }, [activeDistrictId]);

  const analyticsSummary = districtSummariesById[activeDistrictId];

  const allDistrictTotalsConfig = useMemo(() => {
    const map = new Map();
    [...balanceTotalsConfig, ...monthTotalsConfig, ...others].forEach((cfg) => {
      if (!map.has(cfg.key)) map.set(cfg.key, cfg);
    });
    return Array.from(map.values());
  }, []);

  const aggregatedSummary = useMemo(() => {
    const summaries = Object.values(districtSummariesById);
    if (!summaries.length) return null;

    const combinedTotals = allDistrictTotalsConfig.reduce(
      (acc, { key }) => {
        acc[key] = 0;
        return acc;
      },
      {}
    );

    summaries.forEach((summary) => {
      allDistrictTotalsConfig.forEach(({ key }) => {
        const value = Number(summary.column_totals?.[key]) || 0;
        combinedTotals[key] += value;
      });
    });

    return {
      district: 'Andhra Pradesh',
      column_totals: combinedTotals,
    };
  }, [districtSummariesById, allDistrictTotalsConfig]);

  const displaySummary =
    activeDistrictId && analyticsSummary ? analyticsSummary : aggregatedSummary;

  const displayName =
    activeDistrictId && selectedDistrict
      ? selectedDistrict.district
      : displaySummary?.district || 'Select a district';

  const updateSelection = React.useCallback(
    (districtId) => {
      if (!districtId) return;
      setInternalSelectedDistrictId(districtId);
      setDistrictViewMode('currentMonth');

      if (onDistrictSelect) {
        const districtName =
          apDistricts.find((district) => district.id === districtId)
            ?.district ||
          districtSummariesById[districtId]?.district ||
          districtId;
        onDistrictSelect(districtId, districtName);
      }
    },
    [onDistrictSelect, districtSummariesById]
  );

  const inferDistrictFromPath = React.useCallback((pathElement, event) => {
    if (!svgContainerRef.current || !pathElement || !pathElement.getBBox) {
      return null;
    }

      const labels = svgContainerRef.current.querySelectorAll(
        'text.district-label[data-district-id]'
      );
    if (!labels.length) return null;

    const pathRect = pathElement.getBoundingClientRect();
    const cx =
      event?.clientX ?? pathRect.left + (pathRect.width || 0) / 2;
    const cy =
      event?.clientY ?? pathRect.top + (pathRect.height || 0) / 2;

    let closestId = null;
    let closestDist = Infinity;

    labels.forEach((label) => {
      const districtId = label.getAttribute('data-district-id');
      if (!districtId) return;

      const lb = label.getBoundingClientRect();
      const lx = lb.left + (lb.width || 0) / 2;
      const ly = lb.top + (lb.height || 0) / 2;
      const dx = lx - cx;
      const dy = ly - cy;
      const distSq = dx * dx + dy * dy;

      if (distSq < closestDist) {
        closestDist = distSq;
        closestId = districtId;
      }
    });

    return closestId;
  }, []);

  const handleMapClick = React.useCallback(
    (event) => {
      if (!svgContainerRef.current) return;

      const labelNode = event.target.closest('text.district-label');
      const targetPath = labelNode
        ? null
        : event.target.closest('path.district');

      if (!labelNode && !targetPath) return;

      let districtId = null;

      if (labelNode) {
        districtId = resolveDistrictId(labelNode.textContent || '');
      }

      if (!districtId && targetPath) {
        const pathId =
          targetPath.getAttribute('data-district-id') || targetPath.id || '';
        const directMatch = apDistricts.find(
          (district) => district.id === pathId
        );
        if (directMatch) {
          districtId = directMatch.id;
        }
      }

      if (!districtId && targetPath) {
        districtId = inferDistrictFromPath(targetPath, event);
      }

      if (districtId) {
        updateSelection(districtId);
      } else {
        console.log('Clicked SVG element but could not resolve district ID', {
          clickedElement: event.target.tagName,
        });
      }
    },
    [inferDistrictFromPath, updateSelection]
  );

  // Load SVG content (new 26-district map)
  useEffect(() => {
    const svgPaths = [
      '/andhra-pradesh-map.svg',
      '/OCR/andhra-pradesh-map.svg',
    ];

    let currentPathIndex = 0;

    const tryFetch = (path) =>
      fetch(path)
        .then((response) => {
          if (!response.ok) {
            throw new Error(
              `Failed to fetch SVG: ${response.status} ${response.statusText}`
            );
          }
          return response.text();
        })
        .then((text) => {
          if (!text || !text.trim()) {
            throw new Error('Empty SVG content received.');
          }

          const trimmed = text.trim();

          if (
            trimmed.startsWith('<!DOCTYPE') ||
            trimmed.startsWith('<html') ||
            (trimmed.startsWith('<!') && !trimmed.startsWith('<?xml'))
          ) {
            throw new Error('Received HTML instead of SVG.');
          }

          if (!trimmed.startsWith('<svg') && !trimmed.startsWith('<?xml')) {
            throw new Error('Invalid SVG content. Not an SVG.');
          }

          setSvgContent(text);
          setSvgError(null);
        });

    const attemptFetch = () => {
      if (currentPathIndex >= svgPaths.length) {
        setSvgError(
          'Failed to load andhra-pradesh-map.svg from the public folder.'
        );
        return;
      }

      tryFetch(svgPaths[currentPathIndex]).catch((error) => {
        console.warn(
          `Failed to load SVG from ${svgPaths[currentPathIndex]}:`,
          error.message
        );
        currentPathIndex += 1;
        attemptFetch();
      });
    };

    attemptFetch();
  }, []);

  // Process SVG and map district paths and labels after load
  useEffect(() => {
    if (!svgContent) {
      setProcessedSvg(null);
      return;
    }

    try {
      let cleanSvgContent = svgContent.trim();
      if (cleanSvgContent.charCodeAt(0) === 0xfeff) {
        cleanSvgContent = cleanSvgContent.slice(1);
      }

      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(cleanSvgContent, 'image/svg+xml');

      const parserError = svgDoc.querySelector('parsererror');
      if (parserError) {
        const errorText = parserError.textContent || 'Unknown parsing error';
        console.error('SVG parsing error:', errorText);
        setSvgError('SVG parsing error: ' + errorText);
        setProcessedSvg(null);
        return;
      }

      const svgElement = svgDoc.documentElement;

      // Ensure viewBox and sizing classes
      const viewBox =
        svgElement.getAttribute('viewBox') || '0 0 1238.0786 1041.3174';
      svgElement.setAttribute('viewBox', viewBox);
      svgElement.setAttribute(
        'class',
        'w-full h-auto max-w-4xl cursor-pointer'
      );

      // Mark district shapes
      const paths = svgElement.querySelectorAll('path');
      paths.forEach((path) => {
        const d = path.getAttribute('d') || '';
        if (d.length > 100) {
          path.classList.add('district');
          path.setAttribute('data-original-id', path.id || '');
        }
      });

      // Mark district labels
      const textNodes = svgElement.querySelectorAll('text');
      textNodes.forEach((text) => {
        text.classList.add('district-label');
        const existingStyle = text.getAttribute('style') || '';
        text.setAttribute(
          'style',
          `${existingStyle}; cursor: pointer; user-select: none;`
        );

        const districtId = resolveDistrictId(text.textContent || '');
        if (districtId) {
          text.setAttribute('data-district-id', districtId);
          text.setAttribute('role', 'button');
          text.setAttribute('tabindex', '0');
        } else {
          text.removeAttribute('data-district-id');
        }
      });

      // Inject styles for hover/active states
      const style = svgDoc.createElementNS(
        'http://www.w3.org/2000/svg',
        'style'
      );
      style.textContent = `
        .district {
          stroke: #ffffff;
          stroke-width: 2;
          cursor: pointer;
          transition: all 0.25s ease;
        }
        .district:hover {
          stroke: #111827;
          stroke-width: 3;
          filter: drop-shadow(0 0 6px rgba(30, 64, 175, 0.6));
        }
        .district.active {
          stroke: #111827;
          stroke-width: 3.5;
          filter: drop-shadow(0 0 8px rgba(29, 78, 216, 0.85));
        }
        text.district-label {
          cursor: pointer;
        }
      `;
      svgElement.insertBefore(style, svgElement.firstChild);

      const serializer = new XMLSerializer();
      const svgString = serializer.serializeToString(svgElement);
      setProcessedSvg(svgString);
      setSvgError(null);
    } catch (error) {
      console.error('Error processing SVG:', error);
      setSvgError('Error processing SVG: ' + error.message);
      setProcessedSvg(null);
    }
  }, [svgContent]);

  // Highlight active district path
  useEffect(() => {
    if (!svgContainerRef.current || !processedSvg) return;

    const paths = svgContainerRef.current.querySelectorAll('path.district');
    paths.forEach((path) => {
      const pathId =
        path.getAttribute('data-district-id') ||
        path.getAttribute('data-original-id') ||
        path.id;
      if (pathId === activeDistrictId) {
        path.classList.add('active');
      } else {
        path.classList.remove('active');
      }
    });
  }, [activeDistrictId, processedSvg]);

  const hasAnalytics = Boolean(displaySummary);
  const columnTotals = displaySummary?.column_totals || {};

  const activeConfig =
    districtViewMode === 'currentMonth'
      ? monthTotalsConfig
      : balanceTotalsConfig;

  const savingsKeys =
    districtViewMode === 'currentMonth'
      ? ['This Month Savings']
      : ['Total Savings Balance'];
  const newDisbursedKeys = ['New Total'];

  const savingsCards = activeConfig.filter((card) =>
    savingsKeys.includes(card.key)
  );
  const newDisbursedCards = activeConfig.filter((card) =>
    newDisbursedKeys.includes(card.key)
  );
  const loanCards = activeConfig.filter(
    (card) =>
      !savingsKeys.includes(card.key) && !newDisbursedKeys.includes(card.key)
  );

  const middleSectionTitle =
    districtViewMode === 'currentMonth'
      ? 'Loan Repayments & Other Flows'
      : 'Loan Balances & New Loans';

  return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Section */}
        <div className="lg:col-span-2 bg-gradient-to-br from-gray-50 to-gray-100 rounded-3xl shadow-2xl p-6 flex items-center justify-center border-2 border-gray-200 max-h-[640px] overflow-hidden">
        <div
          className="w-full h-auto max-w-4xl flex items-center justify-center cursor-pointer"
          role="img"
          aria-label="Andhra Pradesh District Map"
          onClick={handleMapClick}
        >
          {svgError ? (
            <div className="text-red-600 p-4 text-center max-w-md">
              <p className="font-semibold mb-2">Map Loading Error</p>
              <p className="text-sm">{svgError}</p>
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
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-3xl shadow-2xl p-6 border-2 border-gray-100 max-h-[640px] overflow-y-auto">
        <div className="mb-4">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-3 rounded-2xl">
              <MapPin size={24} className="text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-800">
                District Snapshot
              </h3>
              <p className="text-sm text-gray-500">
                Click a district on the map to view details
              </p>
            </div>
          </div>

          {/* View toggle */}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs font-semibold text-gray-700">
            <div className="flex flex-1 min-w-[200px] items-center justify-between gap-3">
              <label className="flex-1 inline-flex items-center justify-center gap-2 bg-white rounded-xl px-3 py-2 border border-gray-200 cursor-pointer transition hover:border-indigo-300">
                <input
                  type="radio"
                  className="text-indigo-600"
                  checked={districtViewMode === 'currentMonth'}
                  onChange={() => setDistrictViewMode('currentMonth')}
                />
                <span>Current Month</span>
              </label>
              <label className="flex-1 inline-flex items-center justify-center gap-2 bg-white rounded-xl px-3 py-2 border border-gray-200 cursor-pointer transition hover:border-indigo-300">
                <input
                  type="radio"
                  className="text-indigo-600"
                  checked={districtViewMode === 'balances'}
                  onChange={() => setDistrictViewMode('balances')}
                />
                <span>Balances</span>
              </label>
            </div>
          </div>
        </div>

        {hasAnalytics ? (
          <div className="space-y-5">
            {/* District Name */}
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-4 shadow-lg">
              <p className="text-xs text-indigo-100 font-semibold uppercase tracking-wider mb-1">
                Selected District
              </p>
              <p className="text-2xl font-extrabold text-white">
                {displayName}
              </p>
            </div>

            {isAnalyticsLoading && !activeDistrictId && (
              <div className="flex items-center gap-3 bg-indigo-50 border border-indigo-100 text-indigo-700 rounded-2xl px-4 py-3">
                <div className="h-3 w-3 bg-indigo-600 rounded-full animate-ping" />
                <p className="text-sm font-medium">
                  Aggregating statewide totals from the latest Excel upload...
                </p>
              </div>
            )}

            {/* Savings Section */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  Savings
                </h4>
              </div>
              <div className="grid grid-cols-1 gap-3">
                {savingsCards.map(({ label, key, format, accent }) => (
                  <div
                    key={key}
                    className={`bg-gradient-to-br ${accent} rounded-2xl p-3 border shadow-sm`}
                  >
                    <p className="text-[11px] uppercase tracking-wide font-semibold text-gray-600">
                      {label}
                    </p>
                    <p className="text-lg font-extrabold text-gray-900 mt-1">
                      {formatValue(columnTotals[key], format)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Loans Section */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  {middleSectionTitle}
                </h4>
              </div>
              <div className="grid grid-cols-1 gap-3 pr-1">
                {loanCards.map(({ label, key, format, accent }) => (
                  <div
                    key={key}
                    className={`bg-gradient-to-br ${accent} rounded-2xl p-3 border shadow-sm`}
                  >
                    <p className="text-[11px] uppercase tracking-wide font-semibold text-gray-600">
                      {label}
                    </p>
                    <p className="text-lg font-extrabold text-gray-900 mt-1">
                      {formatValue(columnTotals[key], format)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* New Loans Disbursed Section */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                  New Loans Disbursed
                </h4>
              </div>
              <div className="grid grid-cols-1 gap-3">
                {newDisbursedCards.map(({ label, key, format, accent }) => (
                  <div
                    key={key}
                    className={`bg-gradient-to-br ${accent} rounded-2xl p-3 border shadow-sm`}
                  >
                    <p className="text-[11px] uppercase tracking-wide font-semibold text-gray-600">
                      {label}
                    </p>
                    <p className="text-lg font-extrabold text-gray-900 mt-1">
                      {formatValue(columnTotals[key], format)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Others Section */}
            {districtViewMode === 'currentMonth' && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                    Others
                  </h4>
                </div>
                <div className="grid grid-cols-1 gap-3">
                  {others.map(({ label, key, format, accent }) => (
                    <div
                      key={key}
                      className={`bg-gradient-to-br ${accent} rounded-2xl p-3 border shadow-sm`}
                    >
                      <p className="text-[11px] uppercase tracking-wide font-semibold text-gray-600">
                        {label}
                      </p>
                      <p className="text-lg font-extrabold text-gray-900 mt-1">
                        {formatValue(columnTotals[key], format)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Info Message */}
            <div className="bg-gray-100 rounded-xl p-3 border border-gray-200">
              <p className="text-xs text-gray-600 text-center">
                Click on any coloured district or its label on the map to
                update this snapshot. View toggles always reset to{' '}
                <span className="font-semibold">Current Month</span> when you
                change districts.
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <MapPin size={48} className="mx-auto text-gray-300 mb-4" />
            <p className="text-gray-500 text-lg">
              Select a district on the map to view details.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ApMap;


