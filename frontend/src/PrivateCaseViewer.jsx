import { useRef, useState } from 'react'
import './PrivateCaseViewer.css'

const COLORS = { INTC: '#a2f7d0', MU: '#c4a7ff' }

function validateCase(data) {
  if (!data || typeof data !== 'object') {
    throw new Error('The file must contain a case object.')
  }

  if (data.schema_version !== 'private-equity-pair-case-v0.2') {
    throw new Error('Expected the corrected v0.2 private case format.')
  }

  if (data.publication_status !== 'PRIVATE_UNREVIEWED') {
    throw new Error('Expected a PRIVATE_UNREVIEWED case.')
  }

  if (
    data.comparison?.assets?.asset_a !== 'INTC' ||
    data.comparison?.assets?.asset_b !== 'MU'
  ) {
    throw new Error('This first viewer supports the INTC–MU pair only.')
  }

  for (const ticker of ['INTC', 'MU']) {
    const series = data.series?.[ticker]

    if (
      !Array.isArray(series?.date) ||
      !Array.isArray(series?.growth_of_100) ||
      series.date.length < 3 ||
      series.date.length > 10000 ||
      series.date.length !== series.growth_of_100.length
    ) {
      throw new Error(`${ticker}: invalid or oversized chart series.`)
    }

    let previous = ''
    for (let index = 0; index < series.date.length; index += 1) {
      const day = series.date[index]
      const value = series.growth_of_100[index]

      if (
        typeof day !== 'string' ||
        !/^\d{4}-\d{2}-\d{2}$/.test(day) ||
        !Number.isFinite(Date.parse(`${day}T00:00:00Z`)) ||
        (previous && day <= previous) ||
        !Number.isFinite(value) ||
        value <= 0
      ) {
        throw new Error(`${ticker}: invalid dates or chart values.`)
      }
      previous = day
    }
  }

  const datesA = data.series.INTC.date
  const datesB = data.series.MU.date

  if (
    datesA.length !== datesB.length ||
    datesA.some((day, index) => day !== datesB[index])
  ) {
    throw new Error('Chart dates must match exactly in this viewer.')
  }

  for (const ticker of ['INTC', 'MU']) {
    if (Math.abs(data.series[ticker].growth_of_100[0] - 100) > 0.000001) {
      throw new Error(`${ticker}: chart must start at a hypothetical $100.`)
    }
  }

  if (
    !Array.isArray(data.claim_boundaries) ||
    !data.claim_boundaries.every((item) => typeof item === 'string')
  ) {
    throw new Error('Claim boundaries are missing or invalid.')
  }

  if (!data.comparison.surface_similarity || !data.comparison.risk_similarity) {
    throw new Error('Required comparison sections are missing.')
  }

  return data
}

function number(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'Unavailable'
}

function percent(value) {
  return Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : 'Unavailable'
}

function GrowthChart({ series, selected }) {
  const [activeIndex, setActiveIndex] = useState(null)
  const tickers = selected === 'both' ? ['INTC', 'MU'] : [selected]
  const dates = series.INTC.date
  const width = 900
  const height = 340
  const left = 65
  const right = 22
  const top = 25
  const bottom = 45
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const index = activeIndex === null
    ? dates.length - 1
    : Math.min(activeIndex, dates.length - 1)

  let low = 100
  let high = 100

  for (const ticker of tickers) {
    for (const value of series[ticker].growth_of_100) {
      low = Math.min(low, value)
      high = Math.max(high, value)
    }
  }

  const padding = Math.max((high - low) * 0.1, 5)
  low = Math.max(0, low - padding)
  high += padding

  const x = (position) =>
    left + (position / (dates.length - 1)) * plotWidth

  const y = (value) =>
    top + ((high - value) / (high - low)) * plotHeight

  function inspectPointer(event) {
    const svg = event.currentTarget
    const matrix = svg.getScreenCTM()
    if (!matrix) return

    const point = svg.createSVGPoint()
    point.x = event.clientX
    point.y = event.clientY

    const local = point.matrixTransform(matrix.inverse())
    const fraction = Math.max(0, Math.min(1, (local.x - left) / plotWidth))
    setActiveIndex(Math.round(fraction * (dates.length - 1)))
  }

  function money(value) {
    return Number.isFinite(value)
      ? value.toLocaleString('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })
      : 'Unavailable'
  }

  return (
    <div className="private-chart">
      <div className="private-inspector">
        <div className="private-inspector-heading">
          <div>
            <p className="eyebrow">OBSERVATION INSPECTOR</p>
            <p className="private-active-date">{dates[index]}</p>
          </div>
          <p className="private-small">
            Hover over the chart or choose a date below.
          </p>
        </div>

        <div className="private-hover-values">
          {tickers.map((ticker) => (
            <div className="private-hover-asset" key={ticker}>
              <span className="private-asset-name">
                <span
                  className="private-chart-dot"
                  style={{ backgroundColor: COLORS[ticker] }}
                  aria-hidden="true"
                />
                {ticker}
              </span>
              <dl>
                <div>
                  <dt>Adjusted historical price</dt>
                  <dd>{money(series[ticker].adjusted_price?.[index])}</dd>
                </div>
                <div>
                  <dt>Hypothetical $100 value</dt>
                  <dd>{money(series[ticker].growth_of_100[index])}</dd>
                </div>
              </dl>
            </div>
          ))}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Adjusted-price growth rebased to 100 dollars. Use the date slider below for exact values."
        onPointerMove={inspectPointer}
        onPointerDown={inspectPointer}
      >
        {[0, 1, 2, 3, 4].map((step) => {
          const value = low + ((high - low) * step) / 4
          return (
            <g key={step}>
              <line
                x1={left}
                x2={width - right}
                y1={y(value)}
                y2={y(value)}
                stroke="#ffffff14"
              />
              <text x={left - 10} y={y(value) + 4} textAnchor="end">
                ${value.toFixed(0)}
              </text>
            </g>
          )
        })}

        <line
          x1={left}
          x2={width - right}
          y1={y(100)}
          y2={y(100)}
          stroke="#e9d1a1"
          strokeDasharray="5 5"
          opacity="0.7"
        />

        {tickers.map((ticker) => (
          <polyline
            key={ticker}
            points={series[ticker].growth_of_100
              .map((value, position) => `${x(position)},${y(value)}`)
              .join(' ')}
            fill="none"
            stroke={COLORS[ticker]}
            strokeWidth="2.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}

        <line
          x1={x(index)}
          x2={x(index)}
          y1={top}
          y2={height - bottom}
          stroke="#dce3ff"
          strokeDasharray="4 5"
          opacity="0.65"
        />

        {tickers.map((ticker) => (
          <circle
            key={ticker}
            cx={x(index)}
            cy={y(series[ticker].growth_of_100[index])}
            r="5"
            fill={COLORS[ticker]}
            stroke="#080b16"
            strokeWidth="2"
          />
        ))}

        <text x={left} y={height - 12}>{dates[0]}</text>
        <text x={width - right} y={height - 12} textAnchor="end">
          {dates[dates.length - 1]}
        </text>
      </svg>

      <div className="private-date-control">
        <label htmlFor="private-date-slider">
          Inspect observation date: <strong>{dates[index]}</strong>
        </label>
        <input
          id="private-date-slider"
          type="range"
          min="0"
          max={dates.length - 1}
          step="1"
          value={index}
          aria-valuetext={dates[index]}
          onChange={(event) => setActiveIndex(Number(event.target.value))}
        />
      </div>

      <p className="private-small">
        Mint: INTC · Violet: MU. Lines show hypothetical $100 growth,
        not share prices. The inspector separately shows provider-adjusted
        historical prices, not necessarily the unadjusted close quoted that day.
        Values displayed to two decimals. Observations are equally spaced;
        missing sessions are not interpolated.
      </p>
    </div>
  )
}

export default function PrivateCaseViewer() {
  const [caseData, setCaseData] = useState(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState('both')
  const requestId = useRef(0)

  async function openCase(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    const currentRequest = ++requestId.current
    setCaseData(null)
    setError('')

    try {
      if (file.size > 5 * 1024 * 1024) {
        throw new Error('Choose a JSON file smaller than 5 MB.')
      }

      const parsed = validateCase(JSON.parse(await file.text()))
      if (currentRequest !== requestId.current) return
      setCaseData(parsed)
      setSelected('both')
    } catch (problem) {
      if (currentRequest !== requestId.current) return
      setError(problem instanceof Error ? problem.message : 'Cannot read file.')
    }
  }

  function clearCase() {
    requestId.current += 1
    setCaseData(null)
    setError('')
    setSelected('both')
  }

  const comparison = caseData?.comparison
  const surface = comparison?.surface_similarity
  const risk = comparison?.risk_similarity
  const stability = comparison?.relationship_stability

  return (
    <section className="panel private-viewer" aria-labelledby="private-viewer-title">
      <div className="private-heading">
        <div>
          <p className="eyebrow">LOCAL FILE / REAL RESEARCH OUTPUT</p>
          <h2 id="private-viewer-title">Historical evidence explorer</h2>
        </div>
        <span className="hypothesis-badge">Local file · not bundled with site</span>
      </div>

      <p className="private-description">
        Open your corrected INTC–MU case file. This component reads it into
        browser memory without an upload request or persistent storage.
        It checks the file structure, not the scientific validity of its claims.
      </p>

      <div className="private-controls">
        <label className="private-file-label" htmlFor="private-case-file">
          Open private case
        </label>
        <input
          id="private-case-file"
          className="private-file-input"
          type="file"
          accept=".json,application/json"
          onChange={openCase}
        />
        <button
          className="button button-secondary"
          type="button"
          onClick={clearCase}
        >
          Clear loaded case
        </button>
      </div>

      <p className="private-small" role="status">
        {caseData ? 'Case loaded locally. Refresh or clear to remove it.' : 'No case loaded.'}
      </p>

      {error && <p className="private-error" role="alert">{error}</p>}

      {caseData && (
        <>
          <div className="private-case-meta">
            <h3>INTC vs. MU / historical descriptive case</h3>
            <p>
              {comparison.overlap?.returns ?? 'Unavailable'} overlapping returns
              {' · '}
              {String(comparison.overlap?.start ?? '').slice(0, 10)}
              {' → '}
              {String(comparison.overlap?.end ?? '').slice(0, 10)}
            </p>
          </div>

          <div className="private-metrics">
            <div>
              <span>Pearson correlation</span>
              <strong>{number(surface.pearson_correlation)}</strong>
            </div>
            <div>
              <span>Spearman correlation</span>
              <strong>{number(surface.spearman_correlation)}</strong>
            </div>
            <div>
              <span>INTC maximum drawdown</span>
              <strong>{percent(risk.maximum_drawdown_a)}</strong>
            </div>
            <div>
              <span>MU maximum drawdown</span>
              <strong>{percent(risk.maximum_drawdown_b)}</strong>
            </div>
          </div>

          <div className="private-controls" role="group" aria-label="Chart asset selection">
            {[
              ['both', 'Both assets'],
              ['INTC', 'INTC only'],
              ['MU', 'MU only'],
            ].map(([value, label]) => (
              <button
                key={value}
                className="button button-secondary"
                type="button"
                aria-pressed={selected === value}
                onClick={() => setSelected(value)}
              >
                {label}
              </button>
            ))}
          </div>

          <GrowthChart series={caseData.series} selected={selected} />

          <details className="private-details">
            <summary>Relationship stability and interpretation limits</summary>
            <p>
              First-half correlation: {number(stability?.first_half_correlation)}
              {' · '}
              Second-half correlation: {number(stability?.second_half_correlation)}
            </p>
            <p>
              This is a descriptive split, not a formal structural-break test.
              Correlation does not establish causation or a trading opportunity.
            </p>
            <ul>
              {caseData.claim_boundaries.map((boundary, index) => (
                <li key={`${index}-${boundary}`}>{boundary}</li>
              ))}
            </ul>
          </details>
        </>
      )}
    </section>
  )
}
